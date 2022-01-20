import cv2
import os
import numpy as np

pngDirectory = "/Users/alec/Downloads/DAPI/"
files = [f for f in os.listdir(pngDirectory) if os.path.isfile(os.path.join(pngDirectory, f))]
for file in files:
    # Get image path
    imgPath = os.path.join(pngDirectory, file)
    # Load the image
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Get the gaussian threshold
    blur = cv2.GaussianBlur(img,(5,5),0)
    ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # Find the countours in the image, fast method
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Consider the contour arrays only
    contours = contours[0]
    # Start with the first in the list compare subsequent
    xL, yL, wL, hL = cv2.boundingRect(contours[0])
    for c in contours[1:]:
        x, y, w, h  = cv2.boundingRect(c)
        if (w*h) > (wL*hL):
            xL, yL, wL, hL = x, y, w, h
    result = img.copy()
    # Draw bounding box for the largest contour (should be the tissue section)
    cv2.rectangle(result, (xL, yL), (xL+wL, yL+hL), (255, 255, 255), 2)
    # Give a preview
    cv2.imshow("Suggested ROI",result)
    # If we get a spacebar press we are done
    selecting = True
    while selecting:
        k = cv2.waitKey(1) & 0xFF
        if k==32:
            break
        elif k==99:
            custom = img.copy()
            cv2.destroyAllWindows()
            cv2.selectROI("Custom ROI", custom, showCrosshair=False)
    cv2.destroyAllWindows()
