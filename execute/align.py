from aicsimageio import *
import napari
import dask.array as da
import math

img = AICSImage( "/Volumes/T7/M286_slide3_4_rotated.czi", reconstruct_mosaic=False )
img.set_scene(1)
dims = img.dims
rows = []
for tile in range(0, round(dims.M/4)):
    print(f"Processing {tile}/{round(dims.M/4)}...")
    currIndex = -1
    currY = 0
    tileY, tileX = img.get_mosaic_tile_position(tile)
    if currY != tileY:
        currIndex += 1
        rows.append([img.get_image_dask_data("CZYX", M=tile, T=0 )])
    else:
        rows[currIndex].append(img.get_image_dask_data("CZYX", M=tile, T=0 ))

stitched = da.block(rows)
stitched.compute()

viewer = napari.Viewer()
layer = viewer.add_image(stitched)
napari.run()
