from aicsimageio import *
import napari
import numpy as np
import dask.array as da
import cv2

img = AICSImage( "/Volumes/T7/M104-03(41).czi", reconstruct_mosaic=False )

img.set_scene(40)

dims = img.dims

rows = []
currY = 0
currIndex = -1


for tile in range(0, dims.M):
    print(f"Processing {tile}/{dims.M}...", end='\r')
    tileY, tileX = img.get_mosaic_tile_position(tile)
    if tileY > currY + 10:
        currIndex += 1
        currY = tileY
        cell = [img.get_image_dask_data("ZYX", C=2, M=tile, T=0 )]
        rows.append(cell)
    else:
        rows[currIndex].append(img.get_image_dask_data("ZYX", C=2, M=tile, T=0 ))

stitcher = cv2.Stitcher.create()
results = []
forStitch = []
for index in range(0, len( rows )):
    for item in rows[index]:
        toStitch = cv2.cvtColor( np.max(np.asarray(item.compute()), axis=0), cv2.COLOR_GRAY2RGB)
        cv2.imshow("window",toStitch)
        cv2.waitKey(0)
        forStitch.append(toStitch)


results = stitcher.stitch( forStitch )


viewer = napari.Viewer()
layer = viewer.add_image(results[1])
napari.run()
