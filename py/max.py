from tkinter import filedialog
import os
import tifffile as tiff
from skimage.filters import unsharp_mask
import tkinter as tk
import numpy as np
import argparse
import cv2


def process_file(file, outputDirectory, topHat=False, dendrite=False):
    # Update current file
    try:
        print(f"Processing {file}", flush=True)
        img = tiff.imread(file)
        channel_dim = np.argmin(img.shape)
        # Max projection
        img = np.max(img, axis=channel_dim)
        # Get filename stem
        stem = file.split(".")[0]
        # Save the processed image
        cv2.imwrite(f"{outputDirectory}/{stem}.tif", img)

    except Exception as e:
        print(f"Failed to process {file}. Error: {e}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process z-stack images")
    parser.add_argument(
        "-o",
        "--output",
        help="output directory, only use if graphical false",
        default="",
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

    os.chdir(inputDirectory)
    files = os.listdir(".")
    files.sort()
    if len(files) == 0:
        print(1, flush=True)
        print("No files found in input directory", flush=True)
        exit(1)
    # Pass number of files to electron
    print(len(files), flush=True)
    for file in files:
        process_file(file, outputDirectory, args.tophat, args.dendrite)

    print("Done!")
