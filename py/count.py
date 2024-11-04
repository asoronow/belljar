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
    if len(boxes1) == 0 or len(boxes2) == 0:
        return 0
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
    acronym_to_region = {}
    with open(args.structures.strip(), "rb") as f:
        regions = pickle.load(f)
        for k, v in regions.items():
            acronym_to_region[v["acronym"]] = k

    sums = {}
    colocalized = {}
    region_areas = {}
    for i, pName in enumerate(predictionFiles):
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

        # Initialize counts based on args.layers
        sums[pName] = {}
        for c in range(len(predictions)):
            sums[pName][c] = {}
            if args.layers:
                # Include all regions (layers included)
                region_acronyms = set()
                for region_id in regions.keys():
                    region_acronyms.add(regions[region_id]["acronym"])
            else:
                # Exclude layers; use parent regions
                region_acronyms = set()
                for region_id, region_info in regions.items():
                    area_name = region_info["name"]
                    if "layer" not in area_name.lower():
                        region_acronyms.add(region_info["acronym"])
                    else:
                        # Get parent acronym
                        id_path = region_info["id_path"].split("/")
                        if len(id_path) >= 2:
                            parent_id = np.uint32(id_path[-2])
                            parent_acronym = regions[parent_id]["acronym"]
                            region_acronyms.add(parent_acronym)
                        else:
                            region_acronyms.add(region_info["acronym"])
            # Initialize counts to zero
            for acronym in region_acronyms:
                sums[pName][c][acronym] = 0

        all_boxes = {c: [] for c in range(len(predictions))}
        for c, detection in enumerate(predictions):
            counted_boxes = 0
            for box in detection.boxes:
                counted_boxes += 1
                all_boxes[c] += [box]
                x, y, mX, mY = box[0], box[1], box[2], box[3]
                xPos = int((mX - (mX - x) // 2))
                yPos = int((mY - (mY - y) // 2))
                try:
                    atlas_id = annotation_rescaled[yPos, xPos]
                except IndexError:
                    # Resize was in the wrong order
                    annotation_rescaled = resize_image_nearest_neighbor(
                        annotation, predicted_size[::-1]
                    )
                    atlas_id = annotation_rescaled[yPos, xPos]

                region_info = regions[atlas_id]
                acronym = region_info["acronym"]
                if args.layers:
                    # Count the region as is
                    sums[pName][c][acronym] += 1
                else:
                    # Exclude layers
                    area_name = region_info["name"]
                    if "layer" in area_name.lower():
                        id_path = region_info["id_path"].split("/")
                        if len(id_path) >= 2:
                            parent_id = np.uint32(id_path[-2])
                            parent_acronym = regions[parent_id]["acronym"]
                            sums[pName][c][parent_acronym] += 1
                        else:
                            sums[pName][c][acronym] += 1
                    else:
                        sums[pName][c][acronym] += 1

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
        for file, channels in sums.items():
            lines.append([file])
            # Collect region acronyms based on args.layers
            if args.layers:
                all_region_acronyms = set()
                for channel_counts in channels.values():
                    all_region_acronyms.update(channel_counts.keys())
            else:
                all_region_acronyms = set()
                for region_id, region_info in regions.items():
                    area_name = region_info["name"]
                    if "layer" not in area_name.lower():
                        all_region_acronyms.add(region_info["acronym"])
                    else:
                        id_path = region_info["id_path"].split("/")
                        if len(id_path) >= 2:
                            parent_id = np.uint32(id_path[-2])
                            parent_acronym = regions[parent_id]["acronym"]
                            all_region_acronyms.add(parent_acronym)
                        else:
                            all_region_acronyms.add(region_info["acronym"])

            lines.append(
                ["Region Acronym", "Region Name", "Area (px)"]
                + [f"Channel #{c}" for c in range(len(channels))]
            )
            for region in sorted(all_region_acronyms):
                per_channel_counts = []
                for channel in channels:
                    per_channel_counts.append(channels[channel].get(region, 0))
                    if running_counts.get(region, False):
                        running_counts[region] += per_channel_counts[-1]
                    else:
                        running_counts[region] = per_channel_counts[-1]

                # Find name from acronym
                region_id = acronym_to_region.get(region)
                if region_id is None:
                    region_name = "Unknown"
                else:
                    region_name = regions[region_id]["name"]
                region_area = region_areas[file].get(region, 0)
                lines.append(
                    [
                        region,
                        region_name,
                        region_area,
                    ]
                    + per_channel_counts
                )
            lines.append([])

        lines.append(["Totals"])
        lines.append(["Region Acronym", "Region Name", "Count"])
        for region in sorted(running_counts.keys()):
            count = running_counts.get(region, 0)
            region_id = acronym_to_region.get(region)
            if region_id is None:
                region_name = "Unknown"
            else:
                region_name = regions[region_id]["name"]
            lines.append([region, region_name, count])

        lines.append([])
        # Colocalization
        lines.append(["Colocalization Matrix (by Section)"])
        for s, colocal in colocalized.items():
            lines.append([s] + [f"Channel #{c}" for c in range(len(colocal))])
            for c, colocal2 in colocal.items():
                line = [f"Channel #{c}"]
                for c2 in range(len(colocal2)):
                    percent = colocal2.get(c2, 0)
                    line.append(percent)
                lines.append(line)

        writer = csv.writer(resultFile)
        writer.writerows(lines)
    print("Done!", flush=True)