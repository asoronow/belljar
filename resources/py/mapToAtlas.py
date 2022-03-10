import os
import numpy as np
import cv2
from pathlib import Path
from autoencoder import ConvAE2d

# Links in case we should need to redownload these, will not be included
nisslDownloadLink = "http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/ara_nissl/ara_nissil_10.nrrd"
annotationDownloadLink = "http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_10.nrrd"

fileList = os.listdir("../nrrd/png") # path to flat pngs
absolutePaths = [os.path.abspath(p) for p in fileList]
