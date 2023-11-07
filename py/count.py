import argparse
import numpy as np
import os
import csv
import cv2
import pickle
from pathlib import Path
from find_neurons import DetectionResult
from demons import resize_image_nearest_neighbor


def iou(boxA, boxB):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    - boxA: list of [xmin, ymin, xmax, ymax] for the first box.
    - boxB: list of [xmin, ymin, xmax, ymax] for the second box.

    Returns:
    - iou value.
    """
    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection
    inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # Compute the area of both bounding boxes
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Compute the IoU
    iou_value = inter_area / float(boxA_area + boxB_area - inter_area)

    return iou_value


def compute_overlaps(boxes1, boxes2):
    """
    Compute overlaps (IoU) between two sets of boxes.

    Parameters:
    - boxes1: list of bounding boxes. Each box is a list of [xmin, ymin, xmax, ymax].
    - boxes2: list of bounding boxes. Each box is a list of [xmin, ymin, xmax, ymax].

    Returns:
    - overlaps matrix where each element (i, j) is the IoU between boxes1[i] and boxes2[j].
    """
    overlaps = np.zeros((len(boxes1), len(boxes2)))

    for i, box1 in enumerate(boxes1):
        for j, box2 in enumerate(boxes2):
            overlaps[i, j] = iou(box1, box2)

    return overlaps


def percent_colocalized(boxes1, boxes2, threshold=0.5):
    """
    Compute the percentage of boxes in boxes1 that are colocalized with any box in boxes2.

    Parameters:
    - boxes1: list of bounding boxes. Each box is a list of [xmin, ymin, xmax, ymax].
    - boxes2: list of bounding boxes. Each box is a list of [xmin, ymin, xmax, ymax].
    - threshold: Minimum IoU value to consider two boxes as colocalized.

    Returns:
    - Percentage of colocalized boxes.
    """

    overlaps = compute_overlaps(boxes1, boxes2)

    # For each box in boxes1, find the max IoU with any box in boxes2
    max_overlaps = np.max(overlaps, axis=1)

    # Count how many boxes in boxes1 are colocalized with boxes in boxes2
    colocalized_count = np.sum(max_overlaps > threshold)

    return (colocalized_count / len(boxes1)) * 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Integrate cell positions with alignments to count an experiment"
    )
    parser.add_argument(
        "-o",
        "--output",
        help="output directory, only use if graphical false",
        default="",
    )
    parser.add_argument(
        "-p",
        "--predictions",
        help="predictions directory, only use if graphical false",
        default="",
    )
    parser.add_argument(
        "-a",
        "--annotations",
        help="annotations directory, only use if graphical false",
        default="",
    )
    parser.add_argument(
        "-m",
        "--structures",
        help="path to structure map",
        default="../csv/structure_map.pkl",
    )
    parser.add_argument(
        "-l",
        "--layers",
        help="count layers as well",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    prediction_path = Path(args.predictions.strip())
    annotation_path = Path(args.annotations.strip())
    output_path = Path(args.output.strip())

    annotation_files = os.listdir(annotation_path)
    annotation_files = [name for name in annotation_files if name.endswith("pkl")]
    annotation_files.sort()
    print(len(annotation_files) + 1, flush=True)
    predictionFiles = [
        name for name in os.listdir(prediction_path) if name.endswith("pkl")
    ]
    predictionFiles.sort()
    # Reading in regions
    regions = {}
    nameToRegion = {}
    with open(args.structures.strip(), "rb") as f:
        regions = pickle.load(f)
        for k, v in regions.items():
            nameToRegion[v["name"]] = k

    sums = {}
    colocalized = {}
    region_areas = {}
    for i, pName in enumerate(predictionFiles):
        # divide up the results file by section as well
        sums[pName] = {}
        region_areas[pName] = {}
        with open(prediction_path / pName, "rb") as predictionPkl, open(
            annotation_path / annotation_files[i], "rb"
        ) as annotationPkl:
            print(f"Counting {annotation_files[i].split('.')[0]}...", flush=True)
            predictions = pickle.load(predictionPkl)
            predictions = [p for p in predictions]
            annotation = pickle.load(annotationPkl)
            predicted_size = predictions[0].image_dimensions

            # Count the area of each region in the annotation
            annotation_rescaled = resize_image_nearest_neighbor(
                annotation, predicted_size
            )

            unique_ids, counts = np.unique(annotation_rescaled, return_counts=True)
            for unique_id, count in zip(unique_ids, counts):
                name = regions[unique_id]["acronym"]
                id_path = regions[unique_id]["id_path"].split("/")
                if len(id_path) >= 2:
                    parent_id = np.uint32(id_path[-2])
                else:
                    parent_id = unique_id
                parent_name = regions[parent_id]["acronym"]
                region_areas[pName][name] = count

                if not region_areas[pName].get(parent_name, False):
                    region_areas[pName][parent_name] = count
                else:
                    region_areas[pName][parent_name] += count

            all_boxes = {c: [] for c in range(len(predictions))}
            for c, detection in enumerate(predictions):
                sums[pName][c] = {}
                counted_boxes = 0
                for box in detection.boxes:
                    counted_boxes += 1
                    all_boxes[c] += [box]
                    x, y, mX, mY = box[0], box[1], box[2], box[3]
                    xPos = int((mX - (mX - x) // 2))
                    yPos = int((mY - (mY - y) // 2))
                    # draw a circle on the image
                    atlas_id = annotation_rescaled[yPos, xPos]
                    acronym = regions[atlas_id]["acronym"]
                    if args.layers:
                        if sums[pName][c].get(acronym, False):
                            sums[pName][c][acronym] += 1
                        else:
                            sums[pName][c][acronym] = 1
                    else:
                        id_path = regions[atlas_id]["id_path"].split("/")
                        if len(id_path) >= 2:
                            parent_id = np.uint32(id_path[-2])
                        else:
                            parent_id = atlas_id
                        parent_acronym = regions[parent_id]["acronym"]
                        if sums[pName][c].get(parent_acronym, False):
                            sums[pName][c][parent_acronym] += 1
                        else:
                            sums[pName][c][parent_acronym] = 1

            # Compute colocalization
            colocalized[pName] = {}
            local_colocalized = colocalized[pName]
            for c, boxes in all_boxes.items():
                local_colocalized[c] = {}
                for c2, boxes2 in all_boxes.items():
                    local_colocalized[c][c2] = percent_colocalized(boxes, boxes2)

    with open(output_path / "count_results.csv", "w", newline="") as resultFile:
        print("Writing output...", flush=True)
        lines = []
        running_counts = {}
        running_areas = {}
        # Process the sums dictionary to create a unified structure per file
        for file, channels in sums.items():
            lines.append([file])
            all_channel_regions = [channels[channel].keys() for channel in channels]
            all_channel_regions = [
                item for sublist in all_channel_regions for item in sublist
            ]
            lines.append(
                ["Region", "Area (px)"]
                + [f"Channel #{c}" for c in range(len(channels))]
            )
            for region in sorted(all_channel_regions):
                per_channel_counts = []
                for channel in channels:
                    if channels[channel].get(region, False):
                        per_channel_counts.append(channels[channel][region])
                    else:
                        per_channel_counts.append(0)

                    if running_counts.get(region, False):
                        running_counts[region] += per_channel_counts[-1]
                    else:
                        running_counts[region] = per_channel_counts[-1]

                lines.append(
                    [
                        region,
                        region_areas[file][region],
                    ]
                    + per_channel_counts
                )
            lines.append([])

        lines.append(["Totals"])
        lines.append(["Region", "Count"])
        for region, count in sorted(running_counts.items()):
            lines.append([region, count])

        lines.append([])
        # Colocalization
        lines.append(["Colocalization Matrix (by Section)"])
        for s, colocal in colocalized.items():
            lines.append([s] + [f"Channel #{c}" for c in range(len(colocal))])
            for c, colocal2 in colocal.items():
                line = [f"Channel #{c}"]
                for c2, percent in colocal2.items():
                    line.append(percent)
                lines.append(line)

        writer = csv.writer(resultFile)
        writer.writerows(lines)
    print("Done!", flush=True)
