import nrrd
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import interpolation

nData, nHead = nrrd.read('../nrrd/ara_nissl_10.nrrd')
aData, aHead = nrrd.read('../nrrd/annotation_10.nrrd')

print(aHead)


for r in range(-10,11,1):
    nissl_rotatedX = interpolation.rotate(nData[:, :, :], angle=r, axes=(0,2), order=1)
    annotation_rotatedX = interpolation.rotate(aData[:, :, :], angle=r, axes=(0,2), order=1)
    nrrd.write(f'../nrrd/r_nissl_{r}.nrrd', nissl_rotatedX, nHead)
    nrrd.write(f'../nrrd/r_annotation_{r}.nrrd', annotation_rotatedX, aHead)