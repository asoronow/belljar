from tkinter import filedialog, simpledialog
import tifffile as tf
import numpy, os
import tkinter as tk
import cv2

root = tk.Tk()
root.withdraw()

inputDirectory = filedialog.askdirectory(title="Select input directory")
outputDirectory = filedialog.askdirectory(title="Select output directory")
filterSize = simpledialog.askinteger(title="Size of tophat filter", prompt="Provide the size of the filter in px")

def automaticBrightnessAndContrast(image, clipHistPercent=1e-20):
    '''
    Calculates and applies b/c adjustments to an image.
    From: https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape
    '''
    gray = image
    
    # Calculate grayscale histogram
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    histSize = len(hist)
    
    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, histSize):
        accumulator.append(accumulator[index -1] + float(hist[index]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clipHistPercent *= (maximum/100.0)
    clipHistPercent /= 2.0
    
    # Locate left cut
    minimumGray = 0
    while accumulator[minimumGray] < clipHistPercent:
        minimumGray += 1
    
    # Locate right cut
    maximumGray = histSize -1
    while accumulator[maximumGray] >= (maximum - clipHistPercent):
        maximumGray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximumGray - minimumGray)
    beta = -minimumGray * alpha

    auto_result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

os.chdir(inputDirectory)
for file in os.listdir('.'):
    img = tf.imread(file)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (filterSize, filterSize))
    tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    auto, alpha, beta = automaticBrightnessAndContrast(tophat)
    tf.imwrite(f"{outputDirectory}/{file}",  auto)