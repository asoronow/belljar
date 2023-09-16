import cv2
import pickle
import os
from ultralytics import YOLO
from torchvision.ops import nms
import torch
import numpy as np
import argparse
from pathlib import Path
from skimage import morphology

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
    inputDirectory = Path(args.input.strip())
    outputDirectory = Path(args.output.strip())
    tileSize = int(args.tile)
    modelPath = args.model.strip()

    # Pruning
    endings = ["png", "jpg", "jpeg", "tif", "tiff"]
    files = os.listdir(inputDirectory)
    files = [f for f in files if f.split(".")[-1].lower() in endings]
    files.sort()

    print(len(files), flush=True)
    model = YOLO(Path(modelPath))

    for file in files:
        filePath = os.path.join(inputDirectory, file)
        print(f"Processing {file}", flush=True)
        try:
            img = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(15, 15))
            img = clahe.apply(img)
            display = img.copy()
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2RGB)

            selem = morphology.disk(20)
            img = morphology.white_tophat(img, selem)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        except:
            print(f"Error reading {file}", flush=True)
            continue

        # Adjust image
        height, width = img.shape[:2]

        predictionImage = np.zeros((height, width, 3))
        predictionImage[:, :, :] = 255  # make it blank

        num_tiles_x = width // tileSize
        num_tiles_y = height // tileSize

        # Initialize an empty list to hold all predictions
        pred = []
        tiles = []
        overlap = 0.2
        overlap_px = int(tileSize * overlap)  # overlap in pixels

        tile_coords = []

        for i in range(num_tiles_y):
            for j in range(num_tiles_x):
                # Define the coordinates for the current tile
                x1 = max(0, j * tileSize - overlap_px)
                y1 = max(0, i * tileSize - overlap_px)
                x2 = min(img.shape[1], (j + 1) * tileSize + overlap_px)
                y2 = min(img.shape[0], (i + 1) * tileSize + overlap_px)

                tile = img[y1:y2, x1:x2]  # Extract the tile from the image

                tiles.append(tile)
                tile_coords.append((x1, y1, x2, y2))

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cpu":
            # check if MPS is enabled
            try:
                if torch.backends.mps.is_available():
                    dev = "mps"
            except AttributeError:
                pass

        results = model.predict(tiles, conf=0.15, device=dev)
        for i, r in enumerate(results):
            boxes = r.boxes.cpu().numpy()
            scores = boxes.conf
            tile_x1, tile_y1, _, _ = tile_coords[i]

            for j, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].astype(int)

                # Adjust the bounding box coordinates to account for the tile offset
                x1 += tile_x1
                y1 += tile_y1
                x2 += tile_x1
                y2 += tile_y1

                # Save the prediction back
                pred.append((x1, y1, x2, y2, scores[j]))

        # Non-maximum suppression
        if len(pred) == 0:
            pass
        else:
            pred = np.array(pred)
            keep = nms(
                torch.tensor(pred[:, :4], dtype=torch.float32),
                torch.tensor(pred[:, 4], dtype=torch.float32),
                0.4,
            )

            for p in pred[keep]:
                p = p.astype(int)
                cv2.rectangle(display, (p[0], p[1]), (p[2], p[3]), (0, 0, 255), 2)
                cv2.circle(
                    predictionImage,
                    (int((p[0] + p[2]) / 2), int((p[1] + p[3]) / 2)),
                    5,
                    (0, 0, 255),
                    -1,
                )

        # No extension filename
        stripped = file.split(".")[0]

        # Write the prediction image
        cv2.imwrite(
            os.path.join(outputDirectory, f"Dots_{stripped}.png"), predictionImage
        )
        cv2.imwrite(os.path.join(outputDirectory, f"BBoxes_{stripped}.png"), display)

        # Write the raw results to a pkl for later review or reuse
        with open(
            os.path.join(outputDirectory, f"Predictions_{stripped}.pkl"), "wb"
        ) as f:
            # Add the image dimensions as a tuple at the end of the list for reconstruction
            pickle.dump(pred, f)

    print("Done!")
