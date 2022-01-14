import cv2
import os
import numpy as np

pngDirectory = input("Images Directory: ")
files = [f for f in os.listdir(pngDirectory) if os.path.isfile(os.path.isfile(os.path.join(pngDirectory, f)))]

for file in files:
    imgPath = os.path.join(pngDirectory, file)
