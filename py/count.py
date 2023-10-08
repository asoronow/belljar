import argparse
import numpy as np
import os
import csv
import cv2
import pickle
from pathlib import Path
from find_neurons import DetectionResult


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
        "-s",
        "--structures",
        help="structures file",
        default="../csv/structure_tree_safe_2017.csv",
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
    with open(args.structures.strip()) as structureFile:
        structureReader = csv.reader(structureFile, delimiter=",")

        header = next(structureReader)  # skip header
        root = next(structureReader)  # skip atlas root region
        # manually set root, due to weird values
        regions[997] = {"acronym": "undefined", "name": "undefined", "parent": "N/A"}
        regions[0] = {"acronym": "LIW", "name": "Lost in Warp", "parent": "N/A"}
        nameToRegion["undefined"] = 997
        nameToRegion["Lost in Warp"] = 0
        # store all other atlas regions and their linkages
        for row in structureReader:
            regions[int(row[0])] = {
                "acronym": row[3],
                "name": row[2],
                "parent": int(row[8]),
            }
            nameToRegion[row[2]] = int(row[0])

    sums = {}
    colocalized = {}
    region_areas = {}
    for i, pName in enumerate(predictionFiles):
        # divide up the results file by section as well
        sums[annotation_files[i]] = {}
        region_areas[annotation_files[i]] = {}
        local_sums = sums[annotation_files[i]]
        local_region_areas = region_areas[annotation_files[i]]
        with open(prediction_path / pName, "rb") as predictionPkl, open(
            annotation_path / annotation_files[i], "rb"
        ) as annotationPkl:
            print("Counting...", flush=True)
            predictions = pickle.load(predictionPkl)
            predictions = [p for p in predictions]
            annotation = pickle.load(annotationPkl)
            predicted_size = predictions[0].image_dimensions
            predicted_size = (predicted_size[1], predicted_size[0])
            # Count the area of each region in the annotation
            annotation_rescaled = cv2.resize(
                annotation.astype(np.int32),
                predicted_size,
                interpolation=cv2.INTER_NEAREST,
            )
            unique_ids, counts = np.unique(annotation_rescaled, return_counts=True)
            for unique_id, count in zip(unique_ids, counts):
                try:
                    name = regions[unique_id]["name"]
                    parent_id = regions[unique_id]["parent"]
                    parent_name = regions[parent_id]["name"]
                    local_region_areas[name] = local_region_areas.get(name, 0) + count
                    local_region_areas[parent_name] = (
                        local_region_areas.get(parent_name, 0) + count
                    )
                except KeyError:
                    pass

            all_boxes = {c: [] for c in range(len(predictions))}
            for c, detection in enumerate(predictions):
                local_sums[c] = {}
                for box in detection.boxes:
                    all_boxes[c] += [box]
                    x, y, mX, mY = box[0], box[1], box[2], box[3]
                    xPos = int((mX - (mX - x) // 2))
                    yPos = int((mY - (mY - y) // 2))
                    atlas_id = int(annotation_rescaled[yPos, xPos])
                    name = regions[atlas_id]["name"]
                    if "layer" in name.lower():
                        if args.layers:
                            if local_sums[c].get(name, False):
                                local_sums[c][name] += 1
                            else:
                                local_sums[c][name] = 1
                        parent_atlas_id = regions[atlas_id]["parent"]
                        parent_name = regions[parent_atlas_id]["name"]
                        if local_sums[c].get(parent_atlas_id, False):
                            local_sums[c][parent_name] += 1
                        else:
                            local_sums[c][parent_name] = 1
                    else:
                        if local_sums[c].get(name, False):
                            local_sums[c][name] += 1
                        else:
                            local_sums[c][name] = 1

            # Compute colocalization
            colocalized[annotation_files[i]] = {}
            local_colocalized = colocalized[annotation_files[i]]
            for c, boxes in all_boxes.items():
                local_colocalized[c] = {}
                for c2, boxes2 in all_boxes.items():
                    local_colocalized[c][c2] = percent_colocalized(boxes, boxes2)

    with open(output_path / "count_results.csv", "w", newline="") as resultFile:
        print("Writing output...", flush=True)
        lines = []
        running_counts = {}
        running_areas = {}
        sections = list(sums.keys())
        counts = list(sums.values())
        areas = list(region_areas.values())
        for section, counts, areas in zip(sections, counts, areas):
            lines.append([section])
            titles = ["Region", "Acronym", "Area (px^2)"]
            for chan in range(len(counts)):
                titles.append(f"Channel #{chan}")
            lines.append(titles)

            for c, channel_count in counts.items():
                running_counts[c] = {}
                for region, count in channel_count.items():
                    if running_counts[c].get(region, False):
                        running_counts[c][region] += count
                    else:
                        running_counts[c][region] = count

            for name, area in areas.items():
                if running_areas.get(name, False):
                    running_areas[name] += area
                else:
                    running_areas[name] = area

            for region, area in areas.items():
                line = [region, regions[nameToRegion[region]]["acronym"], area]
                per_channel_counts = []
                for chan, channel_count in counts.items():
                    per_channel_counts.append(channel_count.get(region, 0))
                line.extend(per_channel_counts)
                lines.append(line)

            lines.append([])

        lines.append(["Totals"])
        lines.append(
            ["Region", "Acronym", "Area (px^2)"]
            + [f"Channel #{c}" for c in range(len(running_counts))]
        )
        for region, area in running_areas.items():
            per_channel_counts = []
            for chan, count_result in running_counts.items():
                per_channel_counts.append(count_result.get(region, 0))
            line = [
                region,
                regions[nameToRegion[region]]["acronym"],
                area,
                *per_channel_counts,
            ]
            lines.append(line)
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
