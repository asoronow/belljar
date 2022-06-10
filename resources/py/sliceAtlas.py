import nrrd
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import interpolation
# Path to nrrd
nrrdPath = "C:/Users/Alec/.belljar/nrrd"
def buildRotatedAtlases(nisslPath, annotationPath, outputPath):
    '''Constructions the rotated (z-x) atlases for the most common cutting angles'''
    nData, nHead = nrrd.read(nisslPath)
    aData, aHead = nrrd.read(annotationPath)

    for r in range(-10,11,1):
        print(f'Rotating atlas to angle {0}')
        nissl_rotatedX = interpolation.rotate(nData[:, :, :], angle=r, axes=(0,2), order=0)
        annotation_rotatedX = interpolation.rotate(aData[:, :, :], angle=r, axes=(0,2), order=0)
        nrrd.write(str(outputPath) + f'/r_nissl_{r}.nrrd', nissl_rotatedX, nHead)
        nrrd.write(str(outputPath) + f'/r_annotation_{r}.nrrd', annotation_rotatedX, aHead)

def createTrainingSet():
    '''Make the set of all pngs to train the autoencoder'''
    for r in range(-10,11,1):
        print(f'Processing angle {r}')
        data, head = nrrd.read(nrrdPath + f"/r_nissl_{r}.nrrd")
        z, x, y = data.shape
        for slice in range(100,z-100, 1):
            writePath = nrrdPath + "/png_half" + f"/r_nissil_{r}_{slice}.png"
            print(f"Writing slice {slice} to {writePath}")
            image = data[slice, :, :y//2]
            image = cv2.resize(image, (512,512))
            image8 = (image / 256).astype('uint8')
            cv2.imwrite(writePath, image8)

if __name__ == '__main__':
    buildRotatedAtlases()