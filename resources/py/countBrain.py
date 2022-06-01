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
    predictionFiles = os.listdir(args.predictions)
    # Reading in regions
    regions = {}
    with open(args.structures) as structureFile:
        structureReader = csv.reader(structureFile, delimiter=",")
        
        header = next(structureReader) # skip header
        root = next(structureReader) # skip atlas root region

        # store all other atlas regions and their linkages
        for row in structureReader:
            regions[int(row[0])] = {"acronym":row[3], "name":row[2], "parent":int(row[8])}


    sums = {}
    for i, pName in enumerate(predictionFiles):
        # divide up the results file by section as well
        sums[annotationFiles[i][11:]] = {}
        currentSection = sums[annotationFiles[i][11:]]
        with open(args.predictions + pName, 'rb') as predictionPkl:
            prediction = pickle.load(predictionPkl)
            predictedSize = prediction.pop()
            annotation = cv2.imread(args.annotations + annotationFiles[i], -1)
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
    
    print(sums.items())
        