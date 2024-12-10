import cv2
import pickle
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog, simpledialog

root = tk.Tk()
root.withdraw()

simpledialog.Dialog(root, title="Select the input directory")
inputDirectory = filedialog.askdirectory(title="Select input directory")
simpledialog.Dialog(root, title="Select the output directory")
outputDirectory = filedialog.askdirectory(title="Select output directory")

for file in os.listdir(inputDirectory):
    # No extension filename
    stripped = file.replace(file[file.find("."):len(file)], "")
    # Read the raw results from pkl for reuse
    with open(os.path.join(outputDirectory, f"Predictions_{stripped}.pkl"), "rb") as f:
        # List of objects we found
        predictionList = pickle.load(f)
        # File path to image
        filePath = os.path.join(inputDirectory, file)
        # Read it in with cv2
        img = cv2.imread(filePath)
        # TODO: Store dimensions at top of pkl list to prevent need to re-read
        height, width, channels = img.shape
        # Create a blank image for recording predictions
        predictionImage = np.zeros((height, width, 3))
        predictionImage[:, :, :] = 255  # make it blank

        # Make a dot at each object
        for p in predictionList:
            x, y, mX, mY = p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy
            cv2.circle(predictionImage, ((mX - (mX - x)//2),
                       (mY - (mY - y)//2)), 8, (0, 0, 255), -1)
        # Write the prediction image
        cv2.imwrite(os.path.join(outputDirectory,
                    f"Predictions_{stripped}.png"), predictionImage)
