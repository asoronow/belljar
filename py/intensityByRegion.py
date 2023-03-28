import os
import pickle
import argparse
import csv
from pathlib import Path
import cv2
import numpy as np

parser = argparse.ArgumentParser(
    description="Calculate the average intensity of a region in normalized coordinates")

parser.add_argument(
    '-i', '--images', help="input directory for intensity images", default='')
parser.add_argument(
    '-o', '--output', help="output directory for average intensity pkl", default='')
parser.add_argument('-a', '--annotations',
                    help="input directory for annotation pkls", default='')
parser.add_argument('-s', '--structures', help="structures file",
                    default='../csv/structure_tree_safe_2017.csv')
parser.add_argument(
    '-w', '--whole', help="Set True to process a whole brain slice (Default is False)", default=False)
args = parser.parse_args()


if __name__ == '__main__':
    # Read in the intensity images
    intensityPath = args.images.strip()
    intensityFiles = os.listdir(intensityPath)
    intensityFiles.sort()

    # Read the annotation for the images
    annotationPath = args.annotations.strip()
    annotationFiles = os.listdir(annotationPath)
    annotationFiles = [f for f in annotationFiles if f.endswith('.pkl')]
    annotationFiles.sort()

    # Drop .DS_Store files
    if intensityFiles[0] == ".DS_Store":
        intensityFiles.pop(0)

    if annotationFiles[0] == ".DS_Store":
        annotationFiles.pop(0)

    # assert (len(intensityFiles) == len(annotationFile))

    print(2 + len(intensityFiles), flush=True)
    print("Setting up...", flush=True)
    # Read in the regions
    regions = {}
    nameToRegion = {}
    with open(args.structures.strip()) as structureFile:
        structureReader = csv.reader(structureFile, delimiter=",")

        header = next(structureReader)
        root = next(structureReader)
        regions[997] = {"acronym": "undefined",
                        "name": "undefined", "parent": "N/A"}
        regions[0] = {"acronym": "root", "name": "root", "parent": "N/A"}
        nameToRegion["undefined"] = 997
        nameToRegion["root"] = 0
        for row in structureReader:
            regions[int(row[0])] = {"acronym": row[3],
                                    "name": row[2], "parent": int(row[8])}
            nameToRegion[row[3]] = int(row[0])

    for i, iName in enumerate(intensityFiles):
        intensities = {}
        verticies = {}
        # load the image
        intensity = cv2.imread(intensityPath + "/" +
                               iName, cv2.IMREAD_GRAYSCALE)
        # get the image width and height
        height, width = intensity.shape

        # load the annotation
        with open(annotationPath + "/" + annotationFiles[i], 'rb') as f:
            print("Processing " + iName, flush=True)
            annotation = pickle.load(f)
            # get the annotation width and height
            aHeight, aWidth = annotation.shape
            # calculate the scaling factor, minus 200px for annotation padding
            scaleX = width / (aWidth - 200)
            scaleY = height / (aHeight - 200)

            requiredRegions = [
                "VISa",
                "VISal",
                "VISam",
                "VISl",
                "VISli",
                "VISp",
                "VISpl",
                "VISpm",
                "VISpor",
                "VISrl"
            ]

            requiredIds = [nameToRegion[region] for region in requiredRegions]
            # Iterate through the annotation and check if any of the required regions are present
            for i in range(aHeight):
                for j in range(aWidth):
                    # Get to parent acronym
                    regionId = annotation[i, j]
                    if regions.get(regionId, False) == False:
                        regionId = 997

                    if 'layer' in regions[regionId]["name"].lower():
                        regionId = regions[regionId]["parent"]

                    if regionId in requiredIds:
                        # Get the region name
                        regionName = regions[regionId]["acronym"]
                        # Get the region verts
                        imageX = int((j - 100) * scaleX)
                        imageY = int((i - 100) * scaleY)

                        if not verticies.get(regionName, False):
                            verticies[regionName] = []

                        verticies[regionName].append((imageX, imageY))

            # perserve only left points and create mask for extracting intensity
            for region in verticies.keys():
                mask = np.zeros_like(intensity)
                if eval(args.whole.strip()):
                    leftPoints = []
                    for point in verticies[region]:
                        if point[0] < width / 2:
                            leftPoints.append(point)

                    leftHull = cv2.convexHull(np.array(leftPoints))
                    cv2.fillConvexPoly(mask, leftHull, 255)
                else:
                    hull = cv2.convexHull(np.array(verticies[region]))
                    cv2.fillConvexPoly(mask, hull, 255)

                # DEBUG: show mask
                # cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
                # cv2.resizeWindow("mask", 600, 600)
                # resized = cv2.resize(mask, (600, 600))
                # cv2.imshow("mask", mask)
                # cv2.waitKey(3000)

                if not intensities.get(region, False):
                    intensities[region] = {}

                intensityValues = intensity[mask == 255]
                # get the points of the mask'
                points = np.argwhere(mask == 255)

                for i, point in enumerate(points):
                    intensities[region][tuple(point)] = intensityValues[i]

                # DEBUG: show intensity on blank image size of mask
                # blank = np.zeros_like(intensity)
                # blank[mask == 255] = list(intensities[region].values())
                # cv2.namedWindow("intensity", cv2.WINDOW_NORMAL)
                # cv2.resizeWindow("intensity", 600, 600)
                # resized = cv2.resize(
                #     blank, (600, 600))
                # cv2.imshow("intensity", resized)
                # cv2.waitKey(3000)

            # Save the intensity values and the verticies as ROI package pkls
            for region in intensities.keys():
                # split file name
                name = iName.split(".")[0]
                outputPath = Path(args.output.strip() + "/" +
                                  f"{name}_{region}" + ".pkl")
                with open(outputPath, 'wb') as f:
                    pickle.dump(
                        {"roi": intensities[region], "verts": verticies[region], "name": region}, f)

    print("Done!", flush=True)
