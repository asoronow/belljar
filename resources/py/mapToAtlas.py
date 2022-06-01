import os, requests
import numpy as np
import cv2
from pathlib import Path
from sliceAtlas import buildRotatedAtlases
from trainAE import makePredictions
import nrrd
import napari
import argparse
from qtpy.QtWidgets import QPushButton, QProgressBar
from qtpy.QtCore import Qt

parser = argparse.ArgumentParser(description="Map sections to atlas space")
parser.add_argument('-o', '--output', help="output directory, only use if graphical false", default='')
parser.add_argument('-i', '--input', help="input directory, only use if graphical false", default="C:/Users/Alec/.belljar/dapi/subset/")
args = parser.parse_args()

# Links in case we should need to redownload these, will not be included
nisslDownloadLink = "http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/ara_nissl/ara_nissl_10.nrrd"
annotationDownloadLink = "http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_10.nrrd"

# Download files when not present
def downloadFile(url, outpath):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(outpath / local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                f.write(chunk)
    return local_filename

# Warp atlas image roughly onto the DAPI section
def warpToDAPI(atlasImage, dapiImage, annotation):
    '''Takes in a DAPI image and its atlas prediction and warps the atlas to match the section'''
    # TODO: Get height/width of contours, warp to atlas to match height/width of dapi contour
    # TODO: Move atlas contour center to dapi contour center
    # Open the image files.    
    atlasImage = cv2.resize(atlasImage, dapiImage.shape)
    annotation = cv2.resize(annotation, dapiImage.shape, interpolation=cv2.INTER_NEAREST)
    def getMaxContour(image):
        '''Returns the largest contour in an image and its bounding points'''
        # Get the gaussian threshold, otsu method (best automatic results)
        kernel = np.ones((5,5),np.uint8)
        blur = cv2.GaussianBlur(image, (5,5), 0)
        ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # Find the countours in the image, fast method
        contours = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Contours alone
        contours = contours[0]
        # Start with the first in the list compare subsequent
        xL, yL, wL, hL = cv2.boundingRect(contours[0])
        maxC = None
        for c in contours[1:]:
            x, y, w, h  = cv2.boundingRect(c)
            if (w*h) > (wL*hL):
                maxC = c
                xL, yL, wL, hL = x, y, w, h
        
        return maxC, xL, yL, wL, hL
    
    def triangles(points):
        points = np.where(points, points, 1)
        subdiv = cv2.Subdiv2D((*points.min(0), *points.max(0)))
        for pt in points:
            subdiv.insert(tuple(map(int, pt)))
        for pts in subdiv.getTriangleList().reshape(-1, 3, 2):
            yield [np.where(np.all(points == pt, 1))[0][0] for pt in pts]

    def crop(img, pts):
        x, y, w, h = cv2.boundingRect(pts)
        img_cropped = img[y: y + h, x: x + w]
        pts[:, 0] -= x
        pts[:, 1] -= y
        return img_cropped, pts

    def warp(img1, img2, pts1, pts2):
        img2 = img2.copy()
        for indices in triangles(pts1):
            img1_cropped, triangle1 = crop(img1, pts1[indices])
            img2_cropped, triangle2 = crop(img2, pts2[indices])
            transform = cv2.getAffineTransform(np.float32(triangle1), np.float32(triangle2))
            img2_warped = cv2.warpAffine(img1_cropped, transform, img2_cropped.shape[:2][::-1], None, cv2.INTER_NEAREST, cv2.BORDER_TRANSPARENT)
            mask = np.zeros_like(img2_cropped)
            cv2.fillConvexPoly(mask, np.int32(triangle2), (1, 1, 1), 16, 0)
            img2_cropped *= 1 - mask
            img2_cropped += img2_warped * mask
        return img2
    
    dapiContour, dapiX, dapiY, dapiW, dapiH = getMaxContour(dapiImage)

    atlasContour, atlasX, atlasY, atlasW, atlasH = getMaxContour(atlasImage)
    
    dapiContour = np.squeeze(dapiContour)
    atlasContour = np.squeeze(atlasContour)

    atlasRect = np.array([[atlasX, atlasY], [atlasX + atlasW, atlasY], [atlasX + atlasW, atlasY + atlasH], [atlasX, atlasY + atlasH]])
    dapiRect = np.array([[dapiX, dapiY], [dapiX + dapiW, dapiY], [dapiX + dapiW, dapiY + dapiH], [dapiX, dapiY + dapiH]])
    
    atlasResult, annotationResult = np.empty(atlasImage.shape), np.empty(atlasImage.shape)
    try:
        atlasResult = warp(atlasImage, np.zeros(dapiImage.shape), atlasRect, dapiRect)
    except Exception as e:
        print("\n Could not warp atlas image!")
        print(e)
    
    try:
        annotationResult = warp(annotation, np.zeros(dapiImage.shape, dtype="float32"), atlasRect, dapiRect)
    except Exception as e:
        print("\n Could not warp annotations!")
        print(e)

    
    return atlasResult, annotationResult

if __name__ == "__main__":
    # Check if we have the nrrd files
    nrrdPath = Path.home() / ".belljar/nrrd"
    if not nrrdPath.exists():
        # If we don't have what we need, we should grab it from Allen
        os.mkdir(nrrdPath)
        print("Downloading refrence atlas")
        nissl = downloadFile(nisslDownloadLink, nrrdPath)
        annotation = downloadFile(annotationDownloadLink, nrrdPath)
        print("Rotating refrence atlas")
        buildRotatedAtlases(nrrdPath / nissl, nrrdPath / annotation, nrrdPath)

    # Get the file paths
    fileList = os.listdir(args.input)
    absolutePaths = [args.input + p for p in fileList if not p.startswith('.')]
    # Setup the images for analysis
    images = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY) for p in absolutePaths]
    resizedImages = [cv2.resize(im, (512,512)) for im in images]
    # Calculate and get the predictions
    # Predictions dict holds the section numbers for atlas
    predictions, angle = makePredictions(resizedImages, fileList)
    # Load the appropriate atlas
    atlas, atlasHeader = nrrd.read(str(nrrdPath / f"r_nissl_{angle}.nrrd"))
    annotation, annotationHeader = nrrd.read(str(nrrdPath / f"r_annotation_{angle}.nrrd"))
    # Setup the viewer
    viewer = napari.Viewer()
    # Add each layer
    sectionLayer = viewer.add_image(cv2.resize(images[0], (atlas.shape[2]//2,atlas.shape[1])), name="section")
    atlasLayer = viewer.add_image(atlas[:, :, :atlas.shape[2]//2], name="atlas", opacity=0.15)
    # Set the initial slider position
    viewer.dims.set_point(0, predictions[fileList[0]])

    # Track the current section
    currentSection = 0
    # Setup  the napari contorls
    # Button callbacks
    def nextSection():
        '''Move one section forward by crawling file paths'''
        global currentSection, progressBar
        if not currentSection == len(absolutePaths) - 1:
            predictions[fileList[currentSection]] = viewer.dims.current_step[0]
            currentSection += 1
            progressBar.setFormat(f"{currentSection + 1}/{len(absolutePaths)}")
            progressBar.setValue(currentSection + 1)
            sectionLayer.data = cv2.resize(images[currentSection], (atlas.shape[2]//2,atlas.shape[1]))
            viewer.dims.set_point(0, predictions[fileList[currentSection]])
    
    def prevSection():
        '''Move one section backward by crawling file paths'''
        global currentSection, progressBar
        if not currentSection == 0:
            predictions[fileList[currentSection]] = viewer.dims.current_step[0]
            currentSection -= 1
            progressBar.setFormat(f"{currentSection + 1}/{len(absolutePaths)}")
            progressBar.setValue(currentSection + 1)
            progressBar.setValue(currentSection)
            sectionLayer.data = cv2.resize(images[currentSection], (atlas.shape[2]//2,atlas.shape[1]))
            viewer.dims.set_point(0, predictions[fileList[currentSection]])
    
    def finishAlignment():
        '''Save our final updated prediction, perform warps, close'''
        global currentSection
        predictions[fileList[currentSection]] = viewer.dims.current_step[0]
        for i in range(len(absolutePaths)):
            imageName = fileList[i]
            atlasWarp, annoWarp = warpToDAPI((atlas[predictions[imageName], : , :atlas.shape[2]//2]/256).astype('uint8'), 
                                              images[i], 
                                             (annotation[predictions[imageName], : , :annotation.shape[2]//2]).astype('float32')
                                            )
            cv2.imwrite(f"Atlas_{imageName.split('.')[0]}.png", atlasWarp)
            cv2.imwrite(f"Annotation_{imageName.split('.')[0]}.tiff", annoWarp)

        viewer.close()
   
    # Button objects
    nextButton = QPushButton('Next Section')
    backButton = QPushButton('Previous Section')
    doneButton = QPushButton('Done')
    progressBar = QProgressBar(minimum=1, maximum=len(absolutePaths))
    progressBar.setFormat(f"1/{len(absolutePaths)}")
    progressBar.setValue(1)
    progressBar.setAlignment(Qt.AlignCenter)
    # Link callback and objects
    nextButton.clicked.connect(nextSection)
    backButton.clicked.connect(prevSection)
    doneButton.clicked.connect(finishAlignment)
    # Add them to the dock
    viewer.window.add_dock_widget([progressBar, nextButton, backButton, doneButton], name="Bell Jar Controls", area='left')
    # For running from command line
    input("Press ENTER to close")