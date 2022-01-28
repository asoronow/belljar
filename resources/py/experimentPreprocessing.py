import cv2
import os
import numpy as np
import tifffile as tf
from skimage.morphology import white_tophat, disk
from PIL import Image
from PIL import UnidentifiedImageError
from selectiontools import PolygonSelectionTool

class SectionHandler:
    '''
    A class for handling operations on a directory of histilogical section
    images from an experiment. It can optionally restrict the working images
    to single file extension (i.e. png, jpg). It also supports filtering.

    Attributes: directory; the directory this object was generated from
                images; a dictionary of the valid images and their attribbutes
    Methods:

    '''
    def __init__(self, directory, ext=False, allowHidden=False):
        '''Initialize the handler and populate the base filepaths'''
        # Store the directory
        self.directory = directory
        # Dict of image names -> file paths
        self.images = {}
        # Initial inclusion construction
        for f in os.listdir(directory):
            fPath = os.path.join(directory, f)
            if not allowHidden and f[0] == ".":
                continue
            elif os.path.isfile(fPath):
                # We should enforce image validity and extension rules
                flag, details = self.__vetFile(ext, f, fPath)
                if flag:
                    # If we pass then lets append this to our dict
                    self.images[f] = details

    def __vetFile(self, ext, f, fPath):
        '''Enforce file extension and image validity, returns a flag and dict entry for passing images'''
        # Check we match extension if enforced
        if ext != False and f[-len(ext):] == ext:
            # Ensure valid image
            checkup = self.__checkImage(fPath)
            if checkup[0]:
                # Pass params
                return True, {"path": fPath, "width": checkup[1], "height": checkup[2]}
            else:
                # Otherwise fail the check
                return False, None
        # As above, sans extension enforcement
        elif ext == False:
            checkup = self.__checkImage(fPath)
            if checkup[0]:
                return True, {"path": fPath, "width": checkup[1], "height": checkup[2]}
            else:
                return False, None
        else:
            return False, None
    
    def __checkImage(self, path):
        '''Verify this is a valid image, grab dimensions, returns tuple (flag, w, h)'''
        try:
            img = Image.open(path) # try and open with PIL
            width, height = img.size # if we opened we can grab the dims
            return (True, width, height) # return the checkup tuple
        except UnidentifiedImageError as e:
            # Should let the user know we are skipping a bad image, it's likely corrupt and needs fixing
            print(f"Could not verify image at {path}, this means the file is either malformed or this is not an image, skipping!")
            return (False, 0, 0)
            
    def filter(self, name=None, fileSize=None, dimensions=None):
        '''
        Filters files by the given critreon, inclusive, returns and sets new dictionary of files meeting criteria
        Params:
            name: name that must be present in filename
            fileSize: files less than or equal to fileSize in bytes (ex: 10MB === fileSize=10*1e6)
            dimensions: the tuple representing maximum (W,H) of an image, 
                        leave either dimension zero to check one only (i.e., (W,0), (0,H)))
        '''
        maxWidth, maxHeight = dimensions[0], dimensions[1]
        newPaths = {}
        for fName, fPath in self.imagePath.items():
            if name != None:
                if name not in fName:
                    continue
            elif fileSize != None:
                if not os.path.getsize(fPath) <= fileSize:
                    continue
            elif dimensions != None:
                checkFlags = sum([1 if dim > 0 else 0 for dim in dimensions])
                currFlags = 0
                
                if width <= maxWidth:
                    currFlags += 1
                if height <= maxHeight:
                    currFlags += 1
                
                if not currFlags >= checkFlags:
                    continue
            
            newPaths[fName] = fPath 
        
        self.imagePaths = newPaths

    def preprocess(self):
        '''
        Selects tissue section in image and pads out remainder to
        zero intensity. All images are padded to the largest image size
        to enable creation of the final tiff later.
        '''
        maxH, maxW = 0, 0
        for fName, details in sorted(self.images.items()):
            print(f"Processing {fName}...")
            # Check for max dimensions, need this for constructing tiff later
            if (details['width']*details['height']) > (maxW*maxH):
                    maxW, maxH = details['width'], details['height']
            # Load the image
            fPath = details['path']
            img = cv2.imread(fPath)
            # Ensure proper channels
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Get the gaussian threshold, otsu method (best automatic results)
            blur = cv2.GaussianBlur(img,(5,5),0)
            ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # Find the countours in the image, fast method
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Consider the contour arrays only, no hierarchy
            contours = contours[0]
            # Start with the first in the list compare subsequent
            xL, yL, wL, hL = cv2.boundingRect(contours[0])
            maxC = None
            for c in contours[1:]:
                x, y, w, h  = cv2.boundingRect(c)
                if (w*h) > (wL*hL):
                    maxC = c
                    xL, yL, wL, hL = x, y, w, h
            
            # Draw bounding box for the largest contour (should be the tissue section)
            result = img.copy()
            cv2.rectangle(result, (xL, yL), (xL+wL, yL+hL), (255, 255, 255), 2)
            
            # Debug line, shows the contour being detected, can be useful
            # Ex: Image artifacts distorting bounders that needs to be removed
            # cv2.drawContours(result, [maxC], 0, (255,255,255), 3)

            # Give a preview
            cv2.imshow(f"Suggested ROI ({fName})", result)
            cv2.moveWindow(f"Suggested ROI ({fName})", 0,0)
            # If we get a spacebar press we are done, c press draw new poly
            # TODO: Add the logic to pad to max width after we are done with all masks
            # TODO: Make it a tiff
            while True:
                k = cv2.waitKey(1) & 0xFF
                if k==32:
                    mask = np.zeros(img.shape[:2], dtype="uint8")
                    mask[yL:yL+hL,xL:xL+wL] = img[yL:yL+hL,xL:xL+wL]
                    self.images[fName]['mask'] = mask
                    break
                elif k==99:
                    custom = img.copy()
                    cv2.destroyAllWindows()
                    xL, yL, wL, hL = cv2.selectROI(f"Custom ROI ({fName})", custom, showCrosshair=False)
                    mask = np.zeros(img.shape[:2], dtype="uint8")
                    mask[yL:yL+hL,xL:xL+wL] = img[yL:yL+hL,xL:xL+wL]
                    self.images[fName]['mask'] = mask
                    break
        
            cv2.destroyAllWindows()

handle = SectionHandler('Z:\Richard Dickson\R Brains\R13\Exports\DAPI', ext="png")
handle.preprocess()