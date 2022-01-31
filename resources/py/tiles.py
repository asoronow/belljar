from operator import imod
import cv2
import math
import os

# NOTE: Loads of junk in here, testing file tiling for training, data prep
img = cv2.imread("/Users/alec/Downloads/allen_neurons/491609390_59.jpg") # 512x512
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# TODO: Write some math for downsizing image to fit nxm 640px tiles, should not have uneven images
img_shape = img.shape
tile_size = (640,640)
offset = (640, 640)
path = "/Users/alec/Projects/labelGo-Yolov5AutoLabelImg/split"
for i in range(int(math.ceil(img_shape[0]/(offset[1] * 1.0)))):
    for j in range(int(math.ceil(img_shape[1]/(offset[0] * 1.0)))):
        cropped_img = img[offset[1]*i:min(offset[1]*i+tile_size[1], img_shape[0]), offset[0]*j:min(offset[0]*j+tile_size[0], img_shape[1])]
        # Debugging the tiles
        cv2.imwrite(os.path.join(path, "tile_" + str(i) + "_" + str(j) + ".png"), cropped_img)

'''
valid = [file[:-4] for file in os.listdir("/Users/alec/Projects/labelGo-Yolov5AutoLabelImg/split/labels/")]
print(valid)
for f in os.listdir("/Users/alec/Projects/labelGo-Yolov5AutoLabelImg/split/images/"):
    if f[:-4] not in valid:
        os.remove(os.path.join("/Users/alec/Projects/labelGo-Yolov5AutoLabelImg/split/images/", f))
'''
