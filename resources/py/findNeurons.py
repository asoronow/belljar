import cv2, pickle, os
from sahi.model import Yolov5DetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog

root = tk.Tk()
root.withdraw()

modelPath = filedialog.askopenfilename(title="Select the model file") # smnall batch tophat weights
detectionModel = Yolov5DetectionModel(
    model_path=modelPath,
    confidence_threshold=0.5,
    device='cuda:0'
)

inputDirectory = filedialog.askdirectory(title="Select input directory")
outputDirectory = filedialog.askdirectory(title="Select output directory")

for file in os.listdir(inputDirectory):
    if not file.endswith("db") and not file.startswith("."):
        # File path to image
        filePath = os.path.join(inputDirectory, file)
        # Read it in with cv2
        img = cv2.imread(filePath)
        height, width, channels = img.shape
        # Create a blank image for recording predictionsY
        predictionImage = np.zeros((height, width, 3))
        predictionImage[:,:,:] = 255 # make it blank
        # Find cells with SAHI and Model
        result = get_sliced_prediction(img,
                                        detection_model=detectionModel,
                                        slice_height=640,
                                        slice_width=640,
                                        overlap_height_ratio = 0.5,
                                        overlap_width_ratio = 0.5,
                                    )
        # List of objects we found
        predictionList = result.object_prediction_list
        # Make a dot at each object
        for p in predictionList:
            x, y, mX, mY = p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy
            cv2.circle(predictionImage, ((mX - (mX - x)//2),(mY - (mY - y)//2)), 4, (0,0,255), -1)
            cv2.rectangle(img, (x,y), (mX, mY), (255,255,255), 2)
        # No extension filename
        stripped = file.replace(file[file.find("."):len(file)], "")
        print(f"Counted {len(predictionList)} cells")
        # Write the prediction image
        cv2.imwrite(os.path.join(outputDirectory, f"Predictions_{stripped}.png"), predictionImage)
        cv2.imwrite(os.path.join(outputDirectory, f"BBoxes_{stripped}.png"), img)
        # Write the raw results to a pkl for later review or reuse
        with open(os.path.join(outputDirectory, f"Predictions_{stripped}.pkl"), "wb") as f:
            # Add the image dimensions as a tuple at the end of the list for reconstruction
            predictionList.append((height,width))
            pickle.dump(predictionList, f)