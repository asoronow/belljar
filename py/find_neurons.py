import cv2
import pickle
import os
import numpy as np
import argparse
from pathlib import Path
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import tifffile as tiff
import torch


class DetectionResult:
    def __init__(self, boxes, scores, image_dimensions):
        self.boxes = boxes
        self.scores = scores
        self.image_dimensions = image_dimensions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find neurons in images")
    parser.add_argument(
        "-o",
        "--output",
        help="output directory, only use if graphical false",
        default="",
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
        "-n",
        "--multichannel",
        help="specify if multichannel",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    input_dir = Path(args.input.strip())
    output_dir = Path(args.output.strip())
    tile_size = int(args.tile)
    model_path = args.model.strip()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Pruning
    endings = ["png", "jpg", "jpeg", "tif", "tiff"]
    files = os.listdir(input_dir)
    files = [f for f in files if f.split(".")[-1].lower() in endings]
    files.sort()
    print(5 + len(files) * 2, flush=True)  # update users on steps
    print(f"Using device: {device}", flush=True)
    print(f"Using model: {model_path}", flush=True)
    print(f"Using confidence level {float(args.confidence)}", flush=True)
    print(f"Found {len(files)} images", flush=True)
    detection_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=model_path,
        confidence_threshold=float(args.confidence),
        device=device,
    )

    for file in files:
        file_path = os.path.join(input_dir, file)
        stripped, ext = file.split(".")[0], file.split(".")[-1]

        print(f"Running detection on {file}...", flush=True)
        # Try and load the image
        index_order = "F"
        try:
            if ext in ["tif", "tiff"]:
                img = tiff.imread(file_path)
                channels, height, width = img.shape
                index_order = "C"
            else:
                img = cv2.imread(file_path)
                height, width, channels = img.shape
        except:
            print(f"Error reading {file}!", flush=True)
            continue

        # If multichannel, split into individual channels
        split_channels = []
        if args.multichannel:
            for i in range(channels):
                if index_order == "F":
                    split_channels.append(img[:, :, i])
                else:
                    split_channels.append(img[i, :, :])

        predictions = []
        if len(split_channels) > 0:
            for i, chan_img in enumerate(split_channels):
                # convert to BGR
                chan_img = cv2.cvtColor(chan_img, cv2.COLOR_GRAY2BGR)
                # if dtype not uint8, convert
                if chan_img.dtype != np.uint8:
                    chan_img = cv2.normalize(
                        chan_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
                    )
                result = get_sliced_prediction(
                    chan_img,
                    detection_model,
                    slice_height=256,
                    slice_width=256,
                    overlap_height_ratio=0.1,
                    overlap_width_ratio=0.1,
                )

                bboxes = [obj.bbox.to_xyxy() for obj in result.object_prediction_list]
                scores = [obj.score.value for obj in result.object_prediction_list]
                predictions.append(
                    DetectionResult(
                        boxes=bboxes,
                        scores=scores,
                        image_dimensions=(height, width),
                    )
                )
                result.export_visuals(
                    export_dir=output_dir,
                    file_name=f"Boxes_{stripped}_channel_{i}",
                    hide_labels=True,
                    hide_conf=True,
                )
        else:
            # check if image is BGR
            if channels < 3:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            result = get_sliced_prediction(
                img,
                detection_model,
                slice_height=256,
                slice_width=256,
                overlap_height_ratio=0.1,
                overlap_width_ratio=0.1,
            )
            bboxes = [obj.bbox.to_xyxy() for obj in result.object_prediction_list]
            scores = [obj.score.value for obj in result.object_prediction_list]
            predictions = [
                DetectionResult(
                    boxes=bboxes,
                    scores=scores,
                    image_dimensions=(height, width),
                )
            ]
            result.export_visuals(
                export_dir=output_dir,
                file_name=f"Boxes_{stripped}",
                hide_labels=True,
                hide_conf=True,
            )

        with open(output_dir / f"Predictions_{stripped}.pkl", "wb") as f:
            pickle.dump(predictions, f)

    print("Done!", flush=True)
