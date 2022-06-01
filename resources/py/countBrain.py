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
parser.add_argument('-s', '--structures', help="structures file", default='')

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
        next(structureReader) # Skip Line 1
        for row in structureReader:
            regions[row[0]] = row[3]

    for i, pName in enumerate(predictionFiles):
        with open(args.predictions + pName, 'rb') as predictionPkl:
            prediction = pickle.load(predictionPkl)
            predictedSize = prediction.pop()
            annotation = cv2.imread(args.annotations + annotationFiles[i], -1)
            height, width = annotation.shape
            for p in prediction:
                x, y, mX, mY = p.bbox.minx, p.bbox.miny, p.bbox.maxx, p.bbox.maxy
                xPos = (mX - (mX - x)//2)*(width/predictedSize[1])
                yPos = (mY - (mY - y)//2)*(height/predictedSize[0])