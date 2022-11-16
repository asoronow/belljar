import cv2
import pickle
import os
import torch
from sahi.model import Yolov5DetectionModel
from sahi.predict import get_sliced_prediction
import numpy as np
import tkinter as tk
from tkinter import filedialog
import argparse

parser = argparse.ArgumentParser(description="Find neurons in images")
parser.add_argument(
    '-o', '--output', help="output directory, only use if graphical false", default='')
parser.add_argument(
    '-i', '--input', help="input directory, only use if graphical false", default='')
parser.add_argument('-t', '--tile', help="tile size", default=640)
parser.add_argument('-c', '--confidence',
                    help="confidence level for detections", default=0.85)
parser.add_argument('-m', '--model', help="specify model file",
                    default='../models/ancientwizard.pt')
parser.add_argument('-g', '--graphical',
                    help='provides prompts when true', default=True)
args = parser.parse_args()

if args.graphical == True:
    root = tk.Tk()
    root.withdraw()
    inputDirectory = filedialog.askdirectory(title="Select input directory")
    outputDirectory = filedialog.askdirectory(title="Select output directory")
    modelPath = filedialog.askopenfilename(title="Select the model file")
    tileSize = args.tile
else:
    inputDirectory = args.input.strip()
    outputDirectory = args.output.strip()
    tileSize = int(args.tile)
    modelPath = args.model.strip()

detectionModel = Yolov5DetectionModel(
    model_path=modelPath,
    confidence_threshold=float(args.confidence),
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
)
files = os.listdir(inputDirectory)
print(len(files), flush=True)
for file in files:
    if not file.endswith("db") and not file.startswith("."):
        # File path to image
        filePath = os.path.join(inputDirectory, file)
        # Read it in with cv2
        try:
            img = cv2.imread(filePath)
            bbout = cv2.imread(filePath, cv2.IMREAD_COLOR)
            print(f"Processing {file}", flush=True)
            height, width, channels = img.shape
            # Create a blank image for recording predictionsY
            predictionImage = np.zeros((height, width, 3))
            predictionImage[:, :, :] = 255  # make it blank
            # Find cells with SAHI and Model
            result = get_sliced_prediction(img,
                                           detection_model=detectionModel,
                                           slice_height=tileSize,
                                           slice_width=tileSize,
                                           overlap_height_ratio=0.5,
                                           overlap_width_ratio=0.5,
                                           )
            # List of objects we found
            predictionList = result.object_prediction_list
            # Make a dot at each object
            for p in predictionList:
                x, y, mX, mY = p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy
                cv2.circle(predictionImage, ((mX - (mX - x)//2),
                           (mY - (mY - y)//2)), 4, (0, 0, 255), -1)
                cv2.rectangle(bbout, (x, y), (mX, mY), (255, 0, 255), 2)
            # No extension filename
            stripped = file.replace(file[file.find("."):len(file)], "")
            # print(f"Counted {len(predictionList)} cells")
            # Write the prediction image
            cv2.imwrite(os.path.join(outputDirectory,
                        f"Dots_{stripped}.png"), predictionImage)
            cv2.imwrite(os.path.join(outputDirectory,
                        f"BBoxes_{stripped}.png"), bbout)
            # Write the raw results to a pkl for later review or reuse
            with open(os.path.join(outputDirectory, f"Predictions_{stripped}.pkl"), "wb") as f:
                # Add the image dimensions as a tuple at the end of the list for reconstruction
                predictionList.append((height, width))
                pickle.dump(predictionList, f)
        except:
            print("Invalid image!", flush=True)

print("Done!")
