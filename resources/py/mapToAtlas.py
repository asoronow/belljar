import os, requests
import numpy as np
import cv2
from pathlib import Path
from sliceAtlas import buildRotatedAtlases
from trainAE import makePredictions
from torchvision import transforms

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
    pass

if __name__ == "__main__":
    # Check if we have the nrrd files
    nrrdPath = Path.home() / ".belljar/nrrd"
    if nrrdPath.exists():
        fileList = os.listdir(Path.home() / ".belljar/dapi/")[:54]
        absolutePaths = [str(Path.home() / ".belljar/dapi/" / p) for p in fileList]
        images = [cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY), (512,512)) for p in absolutePaths]
        makePredictions(images, fileList)
    else:
        # If we don't have what we need, we should grab it from Allen
        os.mkdir(nrrdPath)
        nissl = downloadFile(nisslDownloadLink, nrrdPath)
        annotation = downloadFile(annotationDownloadLink, nrrdPath)
        buildRotatedAtlases(nrrdPath / nissl, nrrdPath / annotation, nrrdPath)

