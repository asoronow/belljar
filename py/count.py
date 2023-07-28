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
    for i, pName in enumerate(predictionFiles):
        # divide up the results file by section as well
        sums[annotationFiles[i][11:]] = {}
        currentSection = sums[annotationFiles[i][11:]]
        with open(predictionPath / pName, "rb") as predictionPkl, open(
            annotationPath / annotationFiles[i], "rb"
        ) as annotationPkl:
            print("Counting...", flush=True)
            prediction = pickle.load(predictionPkl)
            predictedSize = prediction.pop()
            annotation = pickle.load(annotationPkl)
            height, width = annotation.shape
            for p in prediction:
                x, y, mX, mY = p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy
                xPos = (
                    int((mX - (mX - x) // 2) * ((width) / predictedSize[1]))
                )
                yPos = (
                    int((mY - (mY - y) // 2) * ((height) / predictedSize[0]))
                )
                atlasId = int(annotation[yPos, xPos])
                name = regions[atlasId]["name"]
                if "layer" in name.lower():
                    parent = regions[atlasId]["parent"]
                    name = regions[parent]["name"]
                    if currentSection.get(name, False):
                        currentSection[name] += 1
                    else:
                        currentSection[name] = 1
                else:
                    if currentSection.get(name, False):
                        currentSection[name] += 1
                    else:
                        currentSection[name] = 1

    with open(outputPath / "count_results.csv", "w", newline="") as resultFile:
        print("Writing output...", flush=True)
        lines = []
        runningTotals = {}
        for section, counts in sums.items():
            lines.append([section])
            for r, count in counts.items():
                if runningTotals.get(r, False):
                    runningTotals[r] += count
                else:
                    runningTotals[r] = count

                lines.append([r, regions[nameToRegion[r]]["acronym"], count])
            lines.append([])

        lines.append(["Totals"])
        for r, count in runningTotals.items():
            lines.append([r, regions[nameToRegion[r]]["acronym"], count])
        # Write out the rows
        resultWriter = csv.writer(resultFile)
        resultWriter.writerows(lines)

    print("Done!")
