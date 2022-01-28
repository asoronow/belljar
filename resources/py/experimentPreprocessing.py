import cv2
import os
import numpy as np
from PIL import Image

class SectionHandler:
    '''
    A class for handling operations on a directory of sections.
    It can optionally restrict the working images to a
    single file extension (i.e. png, jpg). It also supports
    filtering the included files.
    '''
    def __init__(self, directory, ext=None, allowHidden=False):
        '''Initialize the handler and populate the base filepaths'''
        # Store the directory
        self.directory = directory
        # Dict of image names -> file paths
        self.imagePaths = {}
        # Initial inclusion construction
        for f in os.listdir(directory):
            fPath = os.path.join(directory, f)
            if not allowHidden and f[0] == ".":
                continue
            elif os.path.isfile(fPath):
                if ext != None and f[:-len(ext)] == ext:
                    self.imagePaths[f] = fPath
                else:
                    self.imagePaths[f] = fPath
            
    def filter(self, name=None, fileSize=None, dimensions=None, inclusive=True):
        '''
        Filters files by a selected critreon, inclusive by default
            name: Check if name is contained in filename
            fileSize: files less than fileSize in bytes
            dimensions: the maximum WxH of an image to be included/excluded, 
                        leave either dimension zero to check one only (i.e., Wx0, 0xH)
        '''
        maxWidth, maxHeight = dimensions.split("x")
        newPaths = {}
        for fName, fPath in self.imagePath.items():
            match = False
            if name != None:
                if name in fName:
                    match = inclusive
            if fileSize != None:
                if os.path.getsize(fPath) <= fileSize:
                    match = inclusive
            if dimensions != None:
                image = Image.open(fPath)
                width, height = image.size
                checkFlags = 0
                currFlags = 0
                if maxHeight > 0:
                    checkFlags += 1
                if maxWidth > 0:
                    checkFlags += 1
                
                if width <= maxWidth:
                    currFlags += 1
                if height <= maxHeight:
                    currFlags += 1
                
                if currFlags >= checkFlags:
                    match = inclusive
            if match:
                newPaths[fName] = fPath 
        self.imagePaths = newPaths

    def preprocess(self):
        '''
        Selects tissue section in image and pads out remainder to
        zero intensity. All images are padded to the largest image size
        to enable creation of the final tiff later.
        '''      
        for fName, fPath in sorted(self.imagePaths.items()):
            # Load the image
            img = cv2.imread(fPath)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Get the gaussian threshold, otsu method (best automatic results)
            blur = cv2.GaussianBlur(img,(5,5),0)
            ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # Find the countours in the image, fast method
            contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Consider the contour arrays only 
            contours = contours[0]
            # Start with the first in the list compare subsequent
            xL, yL, wL, hL = cv2.boundingRect(contours[0])
            for c in contours[1:]:
                x, y, w, h  = cv2.boundingRect(c)
                if (w*h) > (wL*hL):
                    xL, yL, wL, hL = x, y, w, h
            
            # Draw bounding box for the largest contour (should be the tissue section)
            result = img.copy()
            cv2.rectangle(result, (xL, yL), (xL+wL, yL+hL), (255, 255, 255), 2)
            # Give a preview
            cv2.imshow(f"Suggested ROI ({fName})",result)
            # If we get a spacebar press we are done
            selecting = True
            while selecting:
                k = cv2.waitKey(1) & 0xFF
                if k==32:
                    break
                elif k==99:
                    custom = img.copy()
                    cv2.destroyAllWindows()
                    cv2.selectROI(f"Custom ROI ({fName})", custom, showCrosshair=False)
        
            cv2.destroyAllWindows()

handle = SectionHandler('/Users/alec/Downloads/DAPI')
handle.preprocess()