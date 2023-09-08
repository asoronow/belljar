from tkinter import filedialog
import tifffile as tf
import os
from skimage.filters import unsharp_mask
from skimage import filters, morphology
import tkinter as tk
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Process z-stack images")
parser.add_argument(
    "-o", "--output", help="output directory, only use if graphical false", default=""
)
parser.add_argument(
    "-i", "--input", help="input directory, only use if graphical false", default=""
)
parser.add_argument(
    "-g", "--graphical", help="provides prompts when true", default=True
)
parser.add_argument(
    "-d", "--dendrite", help="remove dendrites when true", default=False
)
parser.add_argument(
    "-t", "--tophat", help="apply tophat filter when true", default=False
)
args = parser.parse_args()

if args.graphical == True:
    root = tk.Tk()
    root.withdraw()

    inputDirectory = filedialog.askdirectory(title="Select input directory")
    outputDirectory = filedialog.askdirectory(title="Select output directory")
else:
    inputDirectory = args.input.strip()
    outputDirectory = args.output.strip()


def process_file(file, outputDirectory, topHat=False, dendrite=False):
    # Update current file
    try:
        print(f"Processing {file}", flush=True)
        img = tf.imread(file)

        # Max projection
        img = np.max(img, axis=0)

        # Apply unsharp mask to enhance edges
        img = unsharp_mask(img, radius=1, amount=2)

        # Save the processed image
        tf.imwrite(f"{outputDirectory}/{file}", img)
    except Exception as e:
        print(f"Failed to process {file}. Error: {e}", flush=True)


if __name__ == "__main__":
    os.chdir(inputDirectory)
    files = os.listdir(".")
    files.sort()
    # Pass number of files to electron
    print(len(files), flush=True)
    for file in files:
        process_file(file, outputDirectory, args.tophat, args.dendrite)

    print("Done!")
