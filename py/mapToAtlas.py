import os
import numpy as np
import cv2
import pickle
from pathlib import Path
from trainAE import makePredictions
import nrrd
import csv
import napari
import argparse
from scipy.spatial import distance as dist
from qtpy.QtWidgets import QPushButton, QProgressBar, QCheckBox
from qtpy.QtCore import Qt

parser = argparse.ArgumentParser(description="Map sections to atlas space")
parser.add_argument(
    '-o', '--output', help="output directory, only use if graphical false", default='')
parser.add_argument(
    '-i', '--input', help="input directory, only use if graphical false", default='')
parser.add_argument('-m', "--model", default="../models/predictor_encoder.pt")
parser.add_argument('-e', "--embeds", default="atlasEmbeddings.pkl")
parser.add_argument('-n', "--nrrd",  help="path to nrrd files", default="")
parser.add_argument('-w', "--whole", default=False)
parser.add_argument(
    '-a', "--angle", help="override predicted angle", default=False)
parser.add_argument('-s', '--structures', help="structures file",
                    default='../csv/structure_tree_safe_2017.csv')

args = parser.parse_args()

print(args.angle)

def orderPoints(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype='int64')


# Warp atlas image roughly onto the DAPI section
def warpToDAPI(atlasImage, dapiImage, annotation, shouldDilate=False):
    '''
    Takes in a DAPI image and its atlas prediction and warps the atlas to match the section
    Basis for warp protocol from Ann Zen on Stackoverflow.
    '''
    # Open the image files.
    # Pad the dapi images to ensure no negaitve bounds issues with rotated rects
    dapiImage = np.pad(dapiImage, [(100, 100)], 'constant')
    atlasImage = cv2.resize(atlasImage, dapiImage.shape)
    annotation = cv2.resize(annotation, dapiImage.shape,
                            interpolation=cv2.INTER_NEAREST)

    def dilate(image, kernelSize=21, iterations=3):
        '''Dilate an image'''
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernelSize, kernelSize))
        return cv2.dilate(image, kernel, iterations=iterations)

    def getMaxContour(image, shouldDilate=False):
        '''Returns the largest contour in an image and its bounding points'''
        # Get the gaussian threshold, otsu method (best automatic results)
        if shouldDilate:
            image = dilate(image)

        blur = cv2.GaussianBlur(image, (11, 11), 0)
        ret, thresh = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Find the countours in the image, fast method
        contours = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Contours alone
        contours = contours[0]
        # Start with the first in the list compare subsequentW
        xL, yL, wL, hL = cv2.boundingRect(contours[0])
        maxC = contours[0]
        for c in contours[1:]:
            x, y, w, h = cv2.boundingRect(c)
            if (w*h) > (wL*hL):
                maxC = c
                xL, yL, wL, hL = x, y, w, h

        return maxC, xL, yL, wL, hL

    def triangles(points):
        '''Subdivide a given set of points into triangles'''
        points = np.where(points, points, 1)
        subdiv = cv2.Subdiv2D((*points.min(0), *points.max(0)))
        for pt in points:
            subdiv.insert(tuple(map(int, pt)))
        for pts in subdiv.getTriangleList().reshape(-1, 3, 2):
            yield [np.where(np.all(points == pt, 1))[0][0] for pt in pts]

    def crop(img, pts):
        '''Take just the region of intrest for warping'''
        x, y, w, h = cv2.boundingRect(pts)
        img_cropped = img[y: y + h, x: x + w]
        pts[:, 0] -= x
        pts[:, 1] -= y
        return img_cropped, pts

    def warp(img1, img2, pts1, pts2):
        '''Preform the actual warp by iterating the polygon and warping each triangle'''
        img2 = img2.copy()
        for indices in triangles(pts1):
            img1_cropped, triangle1 = crop(img1, pts1[indices])
            img2_cropped, triangle2 = crop(img2, pts2[indices])
            transform = cv2.getAffineTransform(
                np.float32(triangle1), np.float32(triangle2))
            img2_warped = cv2.warpAffine(
                img1_cropped, transform, img2_cropped.shape[:2][::-1], None, cv2.INTER_NEAREST, cv2.BORDER_TRANSPARENT)
            mask = np.zeros_like(img2_cropped)
            cv2.fillConvexPoly(mask, np.int32(triangle2), (1, 1, 1), 16, 0)
            img2_cropped *= 1 - mask
            img2_cropped += img2_warped * mask
        return img2

    dapiContour, dapiX, dapiY, dapiW, dapiH = getMaxContour(
        dapiImage, shouldDilate)
    atlasContour, atlasX, atlasY, atlasW, atlasH = getMaxContour(atlasImage)

    center, shape, angle = cv2.minAreaRect(dapiContour)
    dapiBox = np.int0(cv2.boxPoints(cv2.minAreaRect(dapiContour)))

    atlasRect = np.array([[atlasX, atlasY], [atlasX + atlasW, atlasY],
                         [atlasX + atlasW, atlasY + atlasH], [atlasX, atlasY + atlasH]])
    dapiRect = np.array([[dapiX, dapiY], [dapiX + dapiW, dapiY],
                        [dapiX + dapiW, dapiY + dapiH], [dapiX, dapiY + dapiH]])

    atlasResult, annotationResult = np.empty(
        dapiImage.shape), np.empty(dapiImage.shape)

    try:
        atlasResult = warp(atlasImage, np.zeros(
            dapiImage.shape), orderPoints(atlasRect), orderPoints(dapiBox))
    except Exception as e:
        pass
        print("\n Could not warp atlas image!")
        print(e)

    try:
        annotationResult = warp(annotation, np.zeros(
            dapiImage.shape, dtype="int32"), orderPoints(atlasRect), orderPoints(dapiBox))
    except Exception as e:
        pass
        print("\n Could not warp annotations!")
        print(e)

    return atlasResult, annotationResult


if __name__ == "__main__":
    # Check if we have the nrrd files
    nrrdPath = Path(args.nrrd.strip())

    # Set if we are using whole or half the brain
    selectionModifier = 2 if not eval(args.whole) else 1

    # Setup path objects
    inputPath = Path(args.input.strip())
    outputPath = Path(args.output.strip())
    # Get the file paths
    fileList = [name for name in os.listdir(inputPath) if os.path.isfile(
        inputPath / name) and not name.startswith('.') and name.endswith('.png')]
    absolutePaths = [str(inputPath / p) for p in fileList]

    # Update the user, next steps are coming
    print(3, flush=True)

    # Setup the images for analysis
    images = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY)
              for p in absolutePaths]
    resizedImages = [cv2.resize(im, (512, 512)) for im in images]
    # Calculate and get the predictions
    # Predictions dict holds the section numbers for atlas
    print("Making predictions...", flush=True)
    predictions, angle, normalizedImages = makePredictions(
        resizedImages, fileList, args.model.strip(), args.embeds.strip(), hemisphere=eval(args.whole))

    # Create a dict to track which sections are seperated
    separated = {}
    for f in fileList:
        separated[f] = False

    # Create a dict to track which sections have been visted
    visited = {}
    for f in fileList:
        visited[f] = False

    # Helper function to adjust predictions of the unvisted sections based on current settings
    def adjustPredictions(predictions, visited, fileList):
        # if any other sections are visited, get the average increase between visted sections
        # and adjust the predictions of the unvisted sections
        # this is to prevent the predictions from being too far off from the visted sections

        if sum([1 for x in visited.values() if x == True]) > 1:
            # Get the average increase between visted sections
            averageIncrease = 0
            for i in range(1, len(fileList)):
                if visited[fileList[i]]:
                    averageIncrease += predictions[fileList[i]
                                                   ] - predictions[fileList[i - 1]]
            averageIncrease /= sum(visited.values())
            # Adjust the predictions of the unvisted sections
            for i in range(len(fileList)):
                if not visited[fileList[i]]:
                    predictions[fileList[i]
                                ] = predictions[fileList[i - 1]] + averageIncrease

        return predictions
    # Load the appropriate atlas
    # Override the angle if needed
    angle = int(args.angle.strip()) if not int(args.angle.strip()) == 99 else angle
    atlas, atlasHeader = nrrd.read(str(nrrdPath / f"r_nissl_{angle}.nrrd"))
    annotation, annotationHeader = nrrd.read(
        str(nrrdPath / f"r_annotation_{angle}.nrrd"))
    print("Awaiting fine tuning...", flush=True)
    # Setup the viewer
    viewer = napari.Viewer()
    # Add each layer
    sectionLayer = viewer.add_image(cv2.resize(
        normalizedImages[0], (atlas.shape[2]//selectionModifier, atlas.shape[1])), name="section")
    atlasLayer = viewer.add_image(
        atlas[:, :, :atlas.shape[2]//selectionModifier], name="atlas", opacity=0.30)
    # Set the initial slider position
    viewer.dims.set_point(0, predictions[fileList[0]])
    # Track the current section
    currentSection = 0
    # Setup  the napari contorls
    # Button callbacks

    def nextSection():
        '''Move one section forward by crawling file paths'''
        global currentSection, progressBar, separatedCheckbox
        if not currentSection == len(normalizedImages) - 1:
            predictions[fileList[currentSection]] = viewer.dims.current_step[0]
            visited[fileList[currentSection]] = True
            if separatedCheckbox.isChecked():
                separated[fileList[currentSection]] = True
            else:
                separated[fileList[currentSection]] = False
            adjustPredictions(predictions, visited, fileList)
            currentSection += 1
            if separated[fileList[currentSection]]:
                separatedCheckbox.setChecked(True)
            else:
                separatedCheckbox.setChecked(False)
            progressBar.setFormat(
                f"{currentSection + 1}/{len(normalizedImages)}")
            progressBar.setValue(currentSection + 1)
            sectionLayer.data = cv2.resize(
                normalizedImages[currentSection], (atlas.shape[2]//selectionModifier, atlas.shape[1]))
            viewer.dims.set_point(0, predictions[fileList[currentSection]])

    def prevSection():
        '''Move one section backward by crawling file paths'''
        global currentSection, progressBar, separatedCheckbox
        if not currentSection == 0:
            predictions[fileList[currentSection]] = viewer.dims.current_step[0]
            visited[fileList[currentSection]] = True
            if separatedCheckbox.isChecked():
                separated[fileList[currentSection]] = True
            else:
                separated[fileList[currentSection]] = False
            adjustPredictions(predictions, visited, fileList)
            currentSection -= 1
            if separated[fileList[currentSection]]:
                separatedCheckbox.setChecked(True)
            else:
                separatedCheckbox.setChecked(False)
            progressBar.setFormat(
                f"{currentSection + 1}/{len(normalizedImages)}")
            progressBar.setValue(currentSection + 1)
            progressBar.setValue(currentSection)
            sectionLayer.data = cv2.resize(
                normalizedImages[currentSection], (atlas.shape[2]//selectionModifier, atlas.shape[1]))
            viewer.dims.set_point(0, predictions[fileList[currentSection]])

    def finishAlignment():
        '''Save our final updated prediction, perform warps, close, also write atlas borders to file'''
        print("Writing output...", flush=True)
        global currentSection, separatedCheckbox
        predictions[fileList[currentSection]] = viewer.dims.current_step[0]
        if separatedCheckbox.isChecked():
            separated[fileList[currentSection]] = True
        else:
            separated[fileList[currentSection]] = False
        # Write the predictions to a file
        for i in range(len(images)):
            imageName = fileList[i]
            print(separated[imageName])
            atlasWarp, annoWarp = warpToDAPI((atlas[predictions[imageName], :, :atlas.shape[2]//selectionModifier]/256).astype('uint8'),
                                             images[i],
                                             (annotation[predictions[imageName], :,
                                              :annotation.shape[2]//selectionModifier]).astype('int32'),
                                             separated[imageName],
                                             )
            cv2.imwrite(
                str(outputPath / f"Atlas_{imageName.split('.')[0]}.png"), atlasWarp)

            with open(str(outputPath / f"Annotation_{imageName.split('.')[0]}.pkl"), "wb") as annoOut:
                pickle.dump(annoWarp, annoOut)

            # Prep regions for saving
            regions = {}
            nameToRegion = {}
            with open(args.structures.strip()) as structureFile:
                structureReader = csv.reader(structureFile, delimiter=",")

                header = next(structureReader)  # skip header
                root = next(structureReader)  # skip atlas root region
                # manually set root, due to weird values
                regions[997] = {"acronym": "undefined",
                                "name": "undefined", "parent": "N/A", "points": []}
                regions[0] = {
                    "acronym": "LIW", "name": "Lost in Warp", "parent": "N/A", "points": []}
                nameToRegion["undefined"] = 997
                nameToRegion["Lost in Warp"] = 0
                # store all other atlas regions and their linkages
                for row in structureReader:
                    regions[int(row[0])] = {
                        "acronym": row[3], "name": row[2], "parent": int(row[8]), "points": []}
                    nameToRegion[row[2]] = int(row[0])

            # Write the atlas borders ontop of dapi image
            # dapi = images[i]
            y, x = annoWarp.shape
            mapImage = np.zeros((y-200, x-200, 3), dtype='uint8')
            for (j, i), area in np.ndenumerate(annoWarp):
                if j > 0 and j < y - 1 and i > 0 and i < x - 1:
                    surroundingPoints = [
                        annoWarp[j, i+1],
                        annoWarp[j+1, i+1],
                        annoWarp[j+1, i-1],
                        annoWarp[j-1, i+1],
                        annoWarp[j+1, i],
                        annoWarp[j, i-1],
                        annoWarp[j-1, i-1],
                        annoWarp[j-1, i]
                    ]
                    if not all(x == area for x in surroundingPoints) and not all(x == 0 for x in surroundingPoints):
                        try:
                            if not all(regions[x]['parent'] == regions[area]['parent'] for x in surroundingPoints):
                                # We are accounting for the padding in the rotation process here
                                # Additonally write this pixel as white
                                mapImage[j-101, i-101] = [255, 255, 255]
                        except:
                            pass
                    try:
                        if all(x == area for x in surroundingPoints):
                            if "layer" in regions[area]['name'].lower():
                                regions[regions[area]['parent']
                                        ]['points'].append((j-101, i-101))
                            else:
                                regions[area]['points'].append((j-101, i-101))
                    except:
                        pass

            # Write the region names
            for region, info in regions.items():
                if info['points'] != [] and region not in [997, 688, 1009, 0]:
                    m = np.mean(info['points'], axis=0).astype(np.int32)
                    try:
                        cv2.putText(mapImage, regions[region]['acronym'], (
                            m[1] - 2, m[0]), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
                    except Exception as e:
                        pass

            cv2.imwrite(
                str(outputPath / f"Map_{imageName.split('.')[0]}.png"), mapImage)

        viewer.close()
        print("Done!", flush=True)
    # Button objects
    nextButton = QPushButton('Next Section')
    backButton = QPushButton('Previous Section')
    doneButton = QPushButton('Done')
    # Add checkbox for seperated
    separatedCheckbox = QCheckBox('Seperated?')
    progressBar = QProgressBar(minimum=1, maximum=len(images))
    progressBar.setFormat(f"1/{len(images)}")
    progressBar.setValue(1)
    progressBar.setAlignment(Qt.AlignCenter)
    # Link callback and objects
    nextButton.clicked.connect(nextSection)
    backButton.clicked.connect(prevSection)
    doneButton.clicked.connect(finishAlignment)
    # Add them to the dock
    viewer.window.add_dock_widget([progressBar, nextButton, backButton,
                                  separatedCheckbox, doneButton], name="Bell Jar Controls", area='left')
    # Start event loop to keep viewer open
    napari.run()