from tkinter import filedialog, simpledialog
import tifffile as tf
import numpy, os
import tkinter as tk
import cv2

root = tk.Tk()
root.withdraw()

inputDirectory = filedialog.askdirectory(title="Select input directory")
outputDirectory = filedialog.askdirectory(title="Select output directory")

def automatic_brightness_and_contrast(image, clip_hist_percent=1e-20):
    gray = image
    
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum/100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size -1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

os.chdir(inputDirectory)
for file in os.listdir('.'):
    img = tf.imread(file)
    filterSize = (25, 25)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, filterSize)
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    auto, alpha, beta = automatic_brightness_and_contrast(tophat)
    tf.imwrite(f"{outputDirectory}/{file}",  auto)