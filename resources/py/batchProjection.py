from tkinter import filedialog, simpledialog
import tifffile as tf
import numpy, os
import tkinter as tk
import cv2

root = tk.Tk()
root.withdraw()

inputDirectory = filedialog.askdirectory(title="Select input directory")
outputDirectory = filedialog.askdirectory(title="Select output directory")

os.chdir(inputDirectory)
for file in os.listdir('.'):
    img = tf.imread(file)
    img = img.max(axis=0)
    tf.imwrite(f"{outputDirectory}/{file}",  img)