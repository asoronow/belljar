from tkinter import filedialog, simpledialog
import tifffile as tf
import os
import tkinter as tk
import cv2
import argparse

parser = argparse.ArgumentParser(description='Process z-stack images')
parser.add_argument('-o', '--output', help="output directory, only use if graphical false", default='')
parser.add_argument('-i', '--input', help="input directory, only use if graphical false", default='')
parser.add_argument('-g', '--graphical', help='provides prompts when true', default=True)
args = parser.parse_args()

if args.graphical:
    root = tk.Tk()
    root.withdraw()

    inputDirectory = filedialog.askdirectory(title="Select input directory")
    outputDirectory = filedialog.askdirectory(title="Select output directory")
else:
    inputDirectory = args.input
    outputDirectory = args.output

os.chdir(inputDirectory)
for file in os.listdir('.'):
    img = tf.imread(file)
    img = img.max(axis=0)
    tf.imwrite(f"{outputDirectory}/{file}",  img)