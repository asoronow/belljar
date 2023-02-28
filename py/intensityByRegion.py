import os
import pickle
import argparse
import csv
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
    annotationFile = os.listdir(annotationPath)
    annotationFile.sort()

    # Drop .DS_Store files
    if intensityFiles[0] == ".DS_Store":
        intensityFiles.pop(0)
    if annotationFile[0] == ".DS_Store":
        annotationFile.pop(0)

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
        with open(annotationPath + "/" + annotationFile[i], 'rb') as f:
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
                    if 'layer' in regions[regionId]["name"].lower():
                        regionId = regions[regionId]["parent"]

                    if regionId in requiredIds:
                        # Get the region name
                        regionName = regions[regionId]["acronym"]
                        # Get the intensity value
                        imageX = int((j - 100) * scaleX)
                        imageY = int((i - 100) * scaleY)
                        intensityValue = intensity[imageY, imageX]
                        intensity[imageY, imageX] = 255
                        # Save the intensity value and the coordinates
                        # Check if the region has been added to the dictionary
                        if not intensities.get(regionName, False):
                            intensities[regionName] = {}

                        if not verticies.get(regionName, False):
                            verticies[regionName] = []

                        intensities[regionName][(
                            imageY, imageX)] = intensityValue
                        verticies[regionName].append((imageX, imageY))

            # Use Cv2 to find the convex hull of each region using its verticies
            # Split them into groups basec on the midpoint of the image
            for region in verticies.keys():
                leftPoints = []
                rightPoints = []
                for point in verticies[region]:
                    if point[0] < width / 2:
                        leftPoints.append(point)
                    else:
                        rightPoints.append(point)

                # Find the convex hull of the left and right points
                leftHull = cv2.convexHull(np.array(leftPoints))
                rightHull = cv2.convexHull(np.array(rightPoints))

                # Now draw a filled polygon using the hull verts
                cv2.fillConvexPoly(intensity, leftHull, 255)
                cv2.fillConvexPoly(intensity, rightHull, 255)

            # Save the intensity values and the verticies as ROI package pkls
            cv2.imshow("intensity", intensity)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            for region in intensities.keys():
                # split file name
                name = iName.split(".")[0]
                with open(args.output + "/" + f"{name}_{region}" + ".pkl", 'wb') as f:
                    pickle.dump(
                        {"roi": intensities[region], "verts": verticies[region], "name": region}, f)
