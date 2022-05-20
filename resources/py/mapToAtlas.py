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
parser.add_argument('-i', '--input', help="input directory, only use if graphical false", default='c:/Users/Alec/.belljar/dapi/')
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
    detector = cv2.SIFT_create()
    dapiKeyPoints, dapiDesc = detector.detectAndCompute(dapiImage, None)
    atlasKeyPoints, atlasDesc = detector.detectAndCompute(atlasImage, None)
    # Find matches
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(dapiDesc,atlasDesc,k=2)
    # store all the good matches as per Lowe's ratio test.
    # good = []
    # for m,n in matches:
    #     if m.distance < 0.7*n.distance:
    #         good.append(m)

    # Find the homography of these two images

    dapi_pts = np.float32([ dapiKeyPoints[m[0].queryIdx].pt for m in matches ])
    atlas_pts = np.float32([ atlasKeyPoints[m[0].trainIdx].pt for m in matches ])
    hom, mask = cv2.findHomography(dapi_pts, atlas_pts, cv2.RANSAC, 5.0)

    # Now actually do the alignment step
    alignedImage = cv2.warpPerspective(atlasImage, hom, (atlasImage.shape[1],atlasImage.shape[0]), flags=cv2.INTER_CUBIC)
    return alignedImage

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
    absolutePaths = [args.input + p for p in fileList[:1]]
    # Setup the images for analysis
    images = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY) for p in absolutePaths]
    resizedImages = [cv2.resize(im, (512,512)) for im in images]
    # Calculate and get the predictions
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
        global currentSection, progressBar
        if not currentSection == len(absolutePaths) - 1:
            predictions[fileList[currentSection]] = viewer.dims.current_step[0]
            currentSection += 1
            progressBar.setFormat(f"{currentSection + 1}/{len(absolutePaths)}")
            progressBar.setValue(currentSection + 1)
            sectionLayer.data = cv2.resize(images[currentSection], (atlas.shape[2]//2,atlas.shape[1]))
            viewer.dims.set_point(0, predictions[fileList[currentSection]])
    def prevSection():
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
        global currentSection
        predictions[fileList[currentSection]] = viewer.dims.current_step[0]
        cv2.imshow("text", (atlas[viewer.dims.current_step[0], : , :atlas.shape[2]//2]/256).astype('uint8'))
        cv2.waitKey(0)
        result = warpToDAPI((atlas[viewer.dims.current_step[0], : , :atlas.shape[2]//2]/256).astype('uint8'), images[currentSection], None)
        cv2.imshow("window", result)
        cv2.waitKey(0)
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