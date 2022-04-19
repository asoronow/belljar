import nrrd
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import interpolation

def buildRotatedAtlases():
    '''Constructions the rotated (z-x) atlases for the most common cutting angles'''
    nData, nHead = nrrd.read('../nrrd/ara_nissl_10.nrrd')
    aData, aHead = nrrd.read('../nrrd/annotation_10.nrrd')

    for r in range(-10,11,1):
        nissl_rotatedX = interpolation.rotate(nData[:, :, :], angle=r, axes=(0,2), order=1)
        annotation_rotatedX = interpolation.rotate(aData[:, :, :], angle=r, axes=(0,2), order=1)
        nrrd.write(f'../nrrd/r_nissl_{r}.nrrd', nissl_rotatedX, nHead)
        nrrd.write(f'../nrrd/r_annotation_{r}.nrrd', annotation_rotatedX, aHead)

def createTrainingSet():
    '''Make the set of all pngs to train the autoencoder'''
    for r in range(-10,11,1):
      data, head = nrrd.read(f"../nrrd/r_nissl_{r}.nrrd")
      z, x, y = data.shape
      for slice in range(100,z-100, 1):
          image = data[slice, :, :y//2]
          image = cv2.resize(image, (512,512))
          image8 = (image / 256).astype('uint8')
          cv2.imwrite(f"../nrrd/png_half/r_nissil_{r}_{slice}.png", image8)

if __name__ == '__main__':
    createTrainingSet()