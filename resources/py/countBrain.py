import argparse
import numpy as np
import os
import csv
import cv2
import pickle

parser = argparse.ArgumentParser(description="Integrate cell positions with alignments to count an experiment")
parser.add_argument('-o', '--output', help="output directory, only use if graphical false", default='')
parser.add_argument('-p', '--predictions', help="input directory, only use if graphical false", default="C:/Users/Alec/Downloads/Predictions/")
parser.add_argument('-a', '--annotations', help="input directory, only use if graphical false", default="C:/Users/Alec/.belljar/dapi/subset/annotation/")
parser.add_argument('-s', '--structures', help="structures file", default='C:/Users/Alec/Desktop/belljar/resources/csv/structure_tree_safe_2017.csv')

args = parser.parse_args()

def countSlice(annotationFile, predictionFile):
    '''Counts all the cells in regions of an annotation file'''
    pass

if __name__ == '__main__':
    annotationFiles = os.listdir(args.annotations)
    predictionFiles = [name for name  in os.listdir(args.predictions) if name.endswith("pkl")]
    # Reading in regions
    regions = {}
    nameToRegion = {}
    with open(args.structures) as structureFile:
        structureReader = csv.reader(structureFile, delimiter=",")
        
        header = next(structureReader) # skip header
        root = next(structureReader) # skip atlas root region
        # manually set root, due to weird values
        regions[997] = {"acronym":"undefined", "name":"undefined", "parent":"N/A"}
        regions[0] = {"acronym":"LIW", "name":"Lost in Warp", "parent":"N/A"}
        # store all other atlas regions and their linkages
        for row in structureReader:
            regions[int(row[0])] = {"acronym":row[3], "name":row[2], "parent":int(row[8])}
            nameToRegion[row[2]] = int(row[0])

    sums = {}
    for i, pName in enumerate(predictionFiles):
        # divide up the results file by section as well
        sums[annotationFiles[i][11:]] = {}
        currentSection = sums[annotationFiles[i][11:]]
        with open(args.predictions + pName, 'rb') as predictionPkl, open(args.annotations + annotationFiles[i], 'rb') as annotationPkl:
            prediction = pickle.load(predictionPkl)
            predictedSize = prediction.pop()
            annotation = pickle.load(annotationPkl)
            height, width = annotation.shape
            for p in prediction:
                x, y, mX, mY = p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy
                xPos = int((mX - (mX - x)//2)*(width/predictedSize[1]))
                yPos = int((mY - (mY - y)//2)*(height/predictedSize[0]))
                atlasId = int(annotation[yPos, xPos])
                name = regions[atlasId]["name"]
                if "layer" in name.lower():
                    parent = regions[atlasId]["parent"]
                    name = regions[parent]["name"]
                    if currentSection.get(name, False):
                        currentSection[name] += 1
                    else:
                        currentSection[name] = 1
    
    with open(args.output + "count_results.csv", "w", newline="") as resultFile:
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
        
        