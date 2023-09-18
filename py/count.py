import argparse
import numpy as np
import os
import csv
import cv2
import pickle
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Integrate cell positions with alignments to count an experiment"
)
parser.add_argument(
    "-o", "--output", help="output directory, only use if graphical false", default=""
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

args = parser.parse_args()

if __name__ == "__main__":
    predictionPath = Path(args.predictions.strip())
    annotationPath = Path(args.annotations.strip())
    outputPath = Path(args.output.strip())

    annotationFiles = os.listdir(annotationPath)
    annotationFiles = [name for name in annotationFiles if name.endswith("pkl")]
    annotationFiles.sort()
    print(len(annotationFiles) + 1, flush=True)
    predictionFiles = [
        name for name in os.listdir(predictionPath) if name.endswith("pkl")
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
    region_areas = {}
    for i, pName in enumerate(predictionFiles):
        # divide up the results file by section as well
        sums[annotationFiles[i][11:]] = {}
        region_areas[annotationFiles[i][11:]] = {}
        local_sums = sums[annotationFiles[i][11:]]
        local_region_areas = region_areas[annotationFiles[i][11:]]
        with open(predictionPath / pName, "rb") as predictionPkl, open(
            annotationPath / annotationFiles[i], "rb"
        ) as annotationPkl:
            print("Counting...", flush=True)
            predictions = pickle.load(predictionPkl)
            predictedSize = predictions.pop()
            predictions = predictions[0].astype(int).tolist()
            annotation = pickle.load(annotationPkl)
            height, width = annotation.shape


            # Count the area of each region in the annotation
            annotation_rescaled = cv2.resize(annotation.astype(np.int32), (predictedSize[1], predictedSize[0]), interpolation=cv2.INTER_NEAREST)
            unique_ids, counts = np.unique(annotation_rescaled, return_counts=True)
            for unique_id, count in zip(unique_ids, counts):
                name = regions[unique_id]["name"]
                parent_id = regions[unique_id]["parent"]
                parent_name = regions[parent_id]["name"] if "layer" in name.lower() else name
                local_region_areas[parent_name] = local_region_areas.get(parent_name, 0) + count

            for p in predictions:
                x, y, mX, mY = p[0], p[1], p[2], p[3]
                xPos = (
                    int((mX - (mX - x) // 2))
                )
                yPos = (
                    int((mY - (mY - y) // 2))
                )
                atlas_id = int(annotation_rescaled[yPos, xPos])
                name = regions[atlas_id]["name"]
                if "layer" in name.lower():
                    atlas_id = regions[atlas_id]["parent"]
                    name = regions[atlas_id]["name"]
                    if local_sums.get(name, False):
                        local_sums[name] += 1
                    else:
                        local_sums[name] = 1
                else:
                    if local_sums.get(name, False):
                        local_sums[name] += 1
                    else:
                        local_sums[name] = 1

    with open(outputPath / "count_results.csv", "w", newline="") as resultFile:
        print("Writing output...", flush=True)
        lines = []
        running_counts = {}
        running_areas = {}
        sections = list(sums.keys())
        counts = list(sums.values())
        areas = list(region_areas.values())
        for section, counts, areas in zip(sections, counts, areas):
            lines.append([section])
            lines.append(["Region", "Acronym", "Count", "Area (px^2)"])
            for r, count in counts.items():
                if running_counts.get(r, False):
                    running_counts[r] += count
                else:
                    running_counts[r] = count
                
                if running_areas.get(r, False):
                    running_areas[r] += areas.get(r, 0)
                else:
                    running_areas[r] = areas.get(r, 0)

                lines.append([r, regions[nameToRegion[r]]["acronym"], count, areas.get(r, 0)])
            lines.append([])

        lines.append(["Totals"])
        lines.append(["Region", "Acronym", "Count", "Area (px^2)"])
        for r, count in running_counts.items():
            lines.append([r, regions[nameToRegion[r]]["acronym"], count, running_areas.get(r, 0)])
        # Write out the rows
        resultWriter = csv.writer(resultFile)
        resultWriter.writerows(lines)

    print("Done!")
