from aicsimageio import *
import napari
import numpy as np
import dask.array as da
import cv2

img = AICSImage( r"C:\Users\imageprocessing\Desktop\M104-03(41).czi", reconstruct_mosaic=False )

img.set_scene(40)

dims = img.dims

rows = []
currY = 0
currIndex = -1


for tile in range(0, dims.M):
    print(f"Processing {tile+1}/{dims.M}...", end='\r')
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
finalImage = []
def cvtToStitch( img ):
    return cv2.cvtColor( np.max( np.asarray( img.compute() ), axis=0 ), cv2.COLOR_GRAY2BGR)

for index in range(0, len( rows )):
    results.append([])
    for item in range(0, len( rows[index] ) - 1):
        imageLeft = cvtToStitch( rows[index][item] )
        imageRight = cvtToStitch( rows[index][item + 1] )
        
        stitched = stitcher.stitch( [imageLeft, imageRight] )

        if stitched[1] != None:
            print("homography")
            if len( results[index] )== 0:
                results[index] = stitched[1]
            else:
                combined = stitcher.stitch([results[index], stitched[1]])
                results[index] = combined[1]
        elif len( results[index] )== 0:
            results[index] = np.concatenate( (imageLeft, imageRight), axis=1 )
        else:
           results[index] = np.concatenate((results[index], imageRight), axis=1)

for row in range(0, len(results) - 1):
    combined = stitcher.stitch((results[row], results[row + 1]))

    if combined[1] != None:
        print("homography")
        if len( finalImage ) == 0:
            finalImage = combined[1]
        else:
            finalImage = stitcher.stitch([finalImage, combined[1]])[1]
    elif len( finalImage ) == 0:
        finalImage = np.concatenate((results[row], results[row + 1]), axis=0)
    else:
        finalImage = np.concatenate((finalImage, results[row + 1]), axis=0)

viewer = napari.Viewer()
layer = viewer.add_image(finalImage)
napari.run()
