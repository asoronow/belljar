from aicsimageio import *
import napari
import dask.array as da

img = AICSImage( "Z:\Matt Jacobs\Images and Data\M Brains\M336\M336_slide2_3-Rotate 2D-02.czi", reconstruct_mosaic=False )
img.set_scene(1)

scene = img.mosaic_dask_data


'''
for tile in range(0, img.dims.M):
    scene.append(img.get_image_dask_data( "CZYX", T=0, M=tile ))

stitched = da.block(scene)
stitched.compute()

viewer = napari.Viewer()
layer = viewer.add_image(stitched)
napari.run()
'''