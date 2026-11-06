import argparse
import os
import sys

import numpy as np
import tifffile as tiff
from section_pairing import check_unique_stems, list_tiffs

# Axis letters tifffile assigns from explicit metadata, and the letters it
# uses when the file does not say what its leading dimensions are (Q for
# shaped arrays without axes, I for a bare page sequence).
KNOWN_AXES = set("ZCSYX")
UNLABELED_AXES = set("QI")


def max_project(img, axes, file):
    """
    Collapse an image to a single YX plane. With axis metadata, every Z, C and S
    axis is max-projected away. Without it, a 2-D image is returned as is and a
    3-D image is projected over its first axis, which must be the smallest.
    """
    if set(axes) <= KNOWN_AXES and "Y" in axes and "X" in axes:
        stack_axes = tuple(i for i, axis in enumerate(axes) if axis not in "YX")
        return np.max(img, axis=stack_axes) if stack_axes else img
    if not set(axes) <= KNOWN_AXES | UNLABELED_AXES:
        raise ValueError(
            f"{os.path.basename(file)} has unsupported axes '{axes}'; "
            "only Z, C and S stack axes can be projected"
        )
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[0] == min(img.shape):
        return np.max(img, axis=0)
    raise ValueError(
        f"{os.path.basename(file)} has shape {img.shape} and no axis metadata; "
        "a stack must be ZYX with Z as the smallest dimension"
    )


def process_file(file, outputDirectory):
    try:
        print(f"Processing {file}", flush=True)
        with tiff.TiffFile(file) as handle:
            series = handle.series[0]
            img = max_project(series.asarray(), series.axes, file)
        stem = os.path.basename(file).split(".")[0]
        tiff.imwrite(f"{outputDirectory}/{stem}.tif", img)

    except (OSError, ValueError) as e:
        sys.exit(f"Failed to process {file}: {e}")


def build_parser():
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
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    if args.graphical is True:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()

        inputDirectory = filedialog.askdirectory(title="Select input directory")
        outputDirectory = filedialog.askdirectory(title="Select output directory")
    else:
        inputDirectory = args.input
        outputDirectory = args.output

    os.chdir(inputDirectory)
    files = list_tiffs(".")
    if len(files) == 0:
        print(1, flush=True)
        print("No files found in input directory", flush=True)
        exit(1)
    try:
        check_unique_stems(files)
    except ValueError as e:
        sys.exit(str(e))
    # Pass number of files to electron
    print(len(files), flush=True)
    for file in files:
        process_file(file, outputDirectory)

    print("Done!")
