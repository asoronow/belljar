from tkinter import filedialog, simpledialog
import tifffile as tf
import os
import tkinter as tk
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Process z-stack images')
parser.add_argument(
    '-o', '--output', help="output directory, only use if graphical false", default='')
parser.add_argument(
    '-i', '--input', help="input directory, only use if graphical false", default='')
parser.add_argument('-g', '--graphical',
                    help='provides prompts when true', default=True)
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
files = os.listdir('.')
# Pass number of files to electron
print(len(files), flush=True)
for file in files:
    try:
        # Update current file
        print(f'Processing {file}', flush=True)
        img = tf.imread(file)
        img = img.max(axis=0)
        tf.imwrite(f"{outputDirectory}/{file}",  img)
    except:
        print("Invalid image!", flush=True)

print('Done!')
