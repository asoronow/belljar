from tkinter import filedialog, simpledialog
import tifffile as tf
import os
import tkinter as tk
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Top hat filter images")
parser.add_argument(
    '-o', '--output', help="output directory, only use if graphical false", default='')
parser.add_argument(
    '-i', '--input', help="input directory, only use if graphical false", default='')
parser.add_argument('-f', '--filter', help="top hat filter size", default='')
parser.add_argument('-c', '--correction',
                    help="gamma correction multiplier", default=1.25)
parser.add_argument('-g', '--graphical',
                    help='provides prompts when true', default=True)
args = parser.parse_args()

if args.graphical == True:
    root = tk.Tk()
    root.withdraw()
    inputDirectory = filedialog.askdirectory(title="Select input directory")
    outputDirectory = filedialog.askdirectory(title="Select output directory")
    filterSize = simpledialog.askinteger(
        title="Size of tophat filter", prompt="Provide the size of the filter in px")

else:
    inputDirectory = args.input.strip()
    outputDirectory = args.output.strip()
    filterSize = int(args.filter.strip())


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


os.chdir(inputDirectory)
files = os.listdir('.')
# Pass number of files to electron
print(len(files), flush=True)
for file in files:
    print(f"Processing {file}", flush=True)
    try:
        img = tf.imread(file)
        # Check if 8-bit
        if img.dtype == 'uint16':
            img = (img / 256).astype('uint8')
        kernel = np.ones((filterSize, filterSize), np.uint8)
        tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        final = adjust_gamma(tophat, float(args.correction))
        tf.imwrite(f"{outputDirectory}/{file}",  final)
    except:
        print("Invalid image!", flush=True)

print("Done!")
