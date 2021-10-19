from aicsimageio import *
import napari
import dask.array as da

'''
img = AICSImage( "/Volumes/T7/M286_slide3_4_rotated.czi", reconstruct_mosaic=False )

data = []
for tile in range(0,3):
    data.append( img.get_image_dask_data("CYX", M=tile, T=0, Z=5) )

stitched = da.block(data)
stitched.compute()
'''

viewer = napari.Viewer()
layer = viewer.add_image(stitched)
napari.run()