import cv2
import pickle
import os
from skimage.measure import label, regionprops
from skimage.filters import threshold_otsu
import numpy as np
import argparse
from pathlib import Path
from skimage.exposure import equalize_adapthist
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import tifffile as tiff
import torch


# Bumped whenever the shape screening semantics change, so results produced
# by different filters can be told apart. Pickles written before versioning
# unpickle with None.
ECCENTRICITY_FILTER_VERSION = 2


class DetectionResult:
    eccentricity_filter_version = None

    def __init__(self, boxes, scores, image_dimensions):
        self.boxes = boxes
        self.scores = scores
        self.image_dimensions = image_dimensions
        self.eccentricity_filter_version = ECCENTRICITY_FILTER_VERSION


def export_bboxes(image, boxes, output_path):
    for box in boxes:
        x1, y1, x2, y2 = [int(b) for b in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(str(output_path), image)


def check_eccentricity(box, threshold, image):
    """Return True when the box should be kept: the largest Otsu-segmented
    blob has eccentricity at or below threshold. A failed check keeps the box."""
    try:
        box = [int(b) for b in box]
        height, width = image.shape[:2]
        y1, y2 = max(box[1] - 5, 0), min(box[3] + 5, height)
        x1, x2 = max(box[0] - 5, 0), min(box[2] + 5, width)
        cell_image = image[y1:y2, x1:x2]
        if len(cell_image.shape) > 2:
            cell_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        mask = cell_image > threshold_otsu(cell_image)

        labeled_mask = label(mask)
        regions = regionprops(labeled_mask)

        if not regions:
            return True

        largest_region = max(regions, key=lambda r: r.area)

        return largest_region.eccentricity <= threshold

    except Exception as e:
        print(f"Eccentricity check failed for box {box}, keeping it. Error: {e}", flush=True)
        return True

def xyxy_to_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def screen_predictions(prediction_objects, area_threshold, eccentricity_threshold=None, image=None, sam_model_path=None):
    """Screen predictions for objects below a certain area"""
    first_pass = [
        obj
        for obj in prediction_objects
        if xyxy_to_area(obj.bbox.to_xyxy()) > area_threshold
    ]
    
    if len(first_pass) < 3:
        return first_pass
    
    # get average area of first pass
    avg_area = sum([xyxy_to_area(obj.bbox.to_xyxy()) for obj in first_pass]) / len( first_pass)
    std_area = np.std([xyxy_to_area(obj.bbox.to_xyxy()) for obj in first_pass])
    
    # second pass. remove objects that are too big
    second_pass = [
        obj
        for obj in first_pass
        if xyxy_to_area(obj.bbox.to_xyxy()) < avg_area + 2 * std_area
    ]

    if eccentricity_threshold is not None:
        try:
            assert image is not None
            second_pass = [
                obj
                for obj in second_pass
                if check_eccentricity(obj.bbox.to_xyxy(), eccentricity_threshold, image)
            ]
        except AssertionError:
            print("Image not provided. Eccentricity screening not performed.")

    return second_pass

def build_parser():
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
    parser.add_argument("-s", "--sam", default="~/.belljar/models/sam_vit_b.pth")
    parser.add_argument("-e", "--eccentricity", help="eccentricity threshold", type=float, default=0.95)
    parser.add_argument(
        "-n",
        "--multichannel",
        help="specify if multichannel",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-a",
        "--area",
        help="area threshold for screening",
        default=200,
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    input_dir = Path(args.input.strip())
    output_dir = Path(args.output.strip())
    tile_size = int(args.tile)
    model_path = args.model.strip()

    # add mps device if available
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"

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
                if len(img.shape) == 3:
                    channels, height, width = img.shape
                    index_order = "C"
                elif len(img.shape) == 2:
                    height, width = img.shape
                    channels = 1
                else:
                    raise Exception("Image has more than 3 dimensions!")
            else:
                img = cv2.imread(file_path)
                height, width, channels = img.shape
        except Exception as e:
            print(f"Error reading {file}!", flush=True)
            print(e, flush=True)
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
                # Check data type
                if chan_img.dtype == np.uint16:
                    chan_img = (chan_img / 256).astype(np.uint8)
                elif chan_img.dtype == np.float32 or chan_img.dtype == np.float64:
                    chan_img = (chan_img * 255).astype(np.uint8)

                # equalize
                chan_img = equalize_adapthist(chan_img, clip_limit=0.01)
                chan_img = (chan_img * 255).astype(np.uint8)
                # convert to BGR
                chan_img = cv2.cvtColor(chan_img, cv2.COLOR_GRAY2BGR)
                result = get_sliced_prediction(
                    chan_img,
                    detection_model,
                    slice_height=tile_size,
                    slice_width=tile_size,
                    overlap_height_ratio=0.1,
                    overlap_width_ratio=0.1,
                )

                predicted_objects = screen_predictions(
                    result.object_prediction_list, 
                    float(args.area.strip()),
                    eccentricity_threshold=args.eccentricity,
                    image=chan_img,
                    sam_model_path=Path(args.sam.strip()).expanduser(),
                    )
                bboxes = [obj.bbox.to_xyxy() for obj in predicted_objects]
                scores = [obj.score.value for obj in predicted_objects]

                predictions.append(
                    DetectionResult(
                        boxes=bboxes,
                        scores=scores,
                        image_dimensions=(height, width),
                    )
                )
                bbox_path = Path(output_dir) / f"BBoxes_{stripped}_{i}.png"
                export_bboxes(chan_img, bboxes, bbox_path)
        else:
            # Check data type
            if img.dtype == np.uint16:
                img = (img / 256).astype(np.uint8)
            elif img.dtype == np.float32 or img.dtype == np.float64:
                img = (img * 255).astype(np.uint8)
            
            img = equalize_adapthist(img, clip_limit=0.01)
            img = (img * 255).astype(np.uint8)

            # convert to BGR
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            result = get_sliced_prediction(
                img,
                detection_model,
                slice_height=tile_size,
                slice_width=tile_size,
                overlap_height_ratio=0.1,
                overlap_width_ratio=0.1,
            )

            predicted_objects = screen_predictions(result.object_prediction_list, 
                                                   float(args.area.strip()), 
                                                   image=img, 
                                                   sam_model_path=Path(args.sam.strip()).expanduser(), 
                                                   eccentricity_threshold=args.eccentricity
                                                   )

            bboxes = [obj.bbox.to_xyxy() for obj in predicted_objects]
            scores = [obj.score.value for obj in predicted_objects]

            predictions = [
                DetectionResult(
                    boxes=bboxes,
                    scores=scores,
                    image_dimensions=(height, width),
                )
            ]
            bbox_path = Path(output_dir) / f"BBoxes_{stripped}.png"
            export_bboxes(img, bboxes, bbox_path)

        with open(output_dir / f"Predictions_{stripped}.pkl", "wb") as f:
            pickle.dump(predictions, f)

    print("Done!", flush=True)
