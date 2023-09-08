import cv2
import tifffile
import pickle
import os
import torch
import yolov5
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Find neurons in images")
parser.add_argument(
    "-o", "--output", help="output directory, only use if graphical false", default=""
)
parser.add_argument(
    "-i", "--input", help="input directory, only use if graphical false", default=""
)
parser.add_argument("-t", "--tile", help="tile size", default=640)
parser.add_argument(
    "-c", "--confidence", help="confidence level for detections", default=0.85
)
parser.add_argument(
    "-m", "--model", help="specify model file", default="../models/ancientwizard.pt"
)
parser.add_argument(
    "-g", "--graphical", help="provides prompts when true", default=True
)
args = parser.parse_args()


if __name__ == "__main__":
    inputDirectory = args.input.strip()
    outputDirectory = args.output.strip()
    tileSize = int(args.tile)
    modelPath = args.model.strip()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Load the YOLOv5 model
    model = yolov5.load(modelPath, device=device)
    model.conf = 0.85

    files = os.listdir(inputDirectory)
    files.sort()
    files = [f for f in files if f.endswith(".tif") or f.endswith(".tiff")]
    print(len(files), flush=True)

    for file in files:
        filePath = os.path.join(inputDirectory, file)
        try:
            img = tifffile.imread(filePath)

            # Convert to 8bit
            if img.dtype != np.uint8:
                # if floating point
                if img.dtype == np.float32 or img.dtype == np.float64:
                    img = img * 255
                    img = img.astype(np.uint8)
                else:
                    img = img.astype(np.uint8)

            bbout = img.copy()
            bbout = cv2.cvtColor(bbout, cv2.COLOR_GRAY2BGR)

            print(f"Processing {file}", flush=True)
            height, width = img.shape[:2]

            predictionImage = np.zeros((height, width, 3))
            predictionImage[:, :, :] = 255  # make it blank

            num_tiles_x = width // tileSize
            num_tiles_y = height // tileSize

            for i in range(num_tiles_y):
                for j in range(num_tiles_x):
                    # Define the coordinates for the current tile
                    x1, y1, x2, y2 = j * tileSize, i * tileSize, (j+1) * tileSize, (i+1) * tileSize
                    tile = img[y1:y2, x1:x2]  # Extract the tile from the image

                    result = model(tile)  # Get predictions for the current tile
                    pred = result.xyxy[0].cpu().numpy()  # Extract predictions
                    
                    # Process the predictions and visualize them on the original image
                    for det in pred:
                        x, y, mX, mY = int(det[0]) + x1, int(det[1]) + y1, int(det[2]) + x1, int(det[3]) + y1
                        cv2.circle(
                            predictionImage,
                            ((mX + x) // 2, (mY + y) // 2),
                            4,
                            (0, 0, 255),
                            -1,
                        )
                        cv2.rectangle(bbout, (x, y), (mX, mY), (255, 0, 255), 2)

            # No extension filename
            stripped = file.split(".")[0]

            # Write the prediction image
            cv2.imwrite(
                os.path.join(outputDirectory, f"Dots_{stripped}.png"), predictionImage
            )
            cv2.imwrite(os.path.join(outputDirectory, f"BBoxes_{stripped}.png"), bbout)
            
            # Write the raw results to a pkl for later review or reuse
            with open(
                os.path.join(outputDirectory, f"Predictions_{stripped}.pkl"), "wb"
            ) as f:
                # Add the image dimensions as a tuple at the end of the list for reconstruction
                pickle.dump(pred, f)
        except Exception as e:
            print(f"Invalid image! {e}", flush=True)

    print("Done!")
