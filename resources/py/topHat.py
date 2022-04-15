from tkinter import filedialog, simpledialog
import tifffile as tf
import numpy, os
import tkinter as tk
import cv2
import numpy as np
root = tk.Tk()
root.withdraw()

inputDirectory = filedialog.askdirectory(title="Select input directory")
outputDirectory = filedialog.askdirectory(title="Select output directory")
filterSize = simpledialog.askinteger(title="Size of tophat filter", prompt="Provide the size of the filter in px")


def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

os.chdir(inputDirectory)
for file in os.listdir('.'):
	print(f"Processing {file}")
	img = tf.imread(file)
	img8 = (img / 256).astype('uint8')
	kernel = np.ones((filterSize,filterSize),np.uint8)
	tophat = cv2.morphologyEx(img8, cv2.MORPH_TOPHAT, kernel)
	final = adjust_gamma(tophat, 1.25)
	tf.imwrite(f"{outputDirectory}/{file}",  final)