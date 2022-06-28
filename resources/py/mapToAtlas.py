import os, requests, math
import numpy as np
import cv2
import pickle
from pathlib import Path
from sliceAtlas import buildRotatedAtlases
from trainAE import makePredictions
import nrrd
import csv
import napari
import argparse
from qtpy.QtWidgets import QPushButton, QProgressBar
from qtpy.QtCore import Qt

parser = argparse.ArgumentParser(description="Map sections to atlas space")
parser.add_argument('-o', '--output', help="output directory, only use if graphical false", default='')
parser.add_argument('-i', '--input', help="input directory, only use if graphical false", default='')
parser.add_argument('-m', "--model", default="../models/predictor_encoder.pt")
parser.add_argument('-e', "--embeds", default="atlasEmbeddings.pkl")
parser.add_argument('-s', '--structures', help="structures file", default='../csv/structure_tree_safe_2017.csv')

args = parser.parse_args()

# Links in case we should need to redownload these, will not be included
nisslDownloadLink = "http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/ara_nissl/ara_nissl_10.nrrd"
annotationDownloadLink = "http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/annotation_10.nrrd"

# Download files when not present
def downloadFile(url, outpath):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(outpath / local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                f.write(chunk)
    return local_filename

def sortClockwise(hull):
    '''Sort the points of a hull clockwise'''
    def key(x):
        atan = math.atan2(x[1], x[0])
        return (atan, x[1]**2+x[0]**2) if atan >= 0 else (2*math.pi + atan, x[0]**2+x[1]**2)

    return np.vstack(sorted(hull, key=key))


# Warp atlas image roughly onto the DAPI section
def warpToDAPI(atlasImage, dapiImage, annotation):
    '''
    Takes in a DAPI image and its atlas prediction and warps the atlas to match the section
    Basis for warp protocol from Ann Zen on Stackoverflow.
    '''
    # Open the image files.
    # Pad the dapi images to ensure no negaitve bounds issues with rotated rects
    dapiImage = np.pad(dapiImage, [(100,100)], 'constant') 
    atlasImage = cv2.resize(atlasImage, dapiImage.shape)
    annotation = cv2.resize(annotation, dapiImage.shape, interpolation=cv2.INTER_NEAREST)
    
    def getMaxContour(image):
        '''Returns the largest contour in an image and its bounding points'''
        # Get the gaussian threshold, otsu method (best automatic results)
        blur = cv2.GaussianBlur(image,(11,11),0)
        ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        # Find the countours in the image, fast method
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Contours alone
        contours = contours[0]
        # Start with the first in the list compare subsequentW
        xL, yL, wL, hL = cv2.boundingRect(contours[0])
        maxC = contours[0]
        for c in contours[1:]:
            x, y, w, h  = cv2.boundingRect(c)
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
            transform = cv2.getAffineTransform(np.float32(triangle1), np.float32(triangle2))
            img2_warped = cv2.warpAffine(img1_cropped, transform, img2_cropped.shape[:2][::-1], None, cv2.INTER_NEAREST, cv2.BORDER_TRANSPARENT)
            mask = np.zeros_like(img2_cropped)
            cv2.fillConvexPoly(mask, np.int32(triangle2), (1, 1, 1), 16, 0)
            img2_cropped *= 1 - mask
            img2_cropped += img2_warped * mask
        return img2
    

    dapiContour, dapiX, dapiY, dapiW, dapiH = getMaxContour(dapiImage)
    atlasContour, atlasX, atlasY, atlasW, atlasH = getMaxContour(atlasImage)
    
    center, shape, angle = cv2.minAreaRect(dapiContour)
    dapiBox = np.int0(cv2.boxPoints(cv2.minAreaRect(dapiContour)))

    atlasRect = np.array([[atlasX, atlasY], [atlasX + atlasW, atlasY], [atlasX + atlasW, atlasY + atlasH], [atlasX, atlasY + atlasH]])
    dapiRect = np.array([[dapiX, dapiY], [dapiX + dapiW, dapiY], [dapiX + dapiW, dapiY + dapiH], [dapiX, dapiY + dapiH]])
    
    atlasResult, annotationResult = np.empty(dapiImage.shape), np.empty(dapiImage.shape)
    
    try:
        if angle < 45:
            atlasResult = warp(atlasImage, np.zeros(dapiImage.shape), sortClockwise(atlasRect), sortClockwise(dapiBox))
        else:
            atlasResult = warp(atlasImage, np.zeros(dapiImage.shape), atlasRect, dapiBox)
    except Exception as e:
        pass
        # print("\n Could not warp atlas image!")
        # print(e)
    
    try:
        if angle < 45:
            annotationResult = warp(annotation, np.zeros(dapiImage.shape, dtype="int32"), sortClockwise(atlasRect), sortClockwise(dapiBox))
        else:
            annotationResult = warp(annotation, np.zeros(dapiImage.shape, dtype="int32"), atlasRect, dapiBox)
    except Exception as e:
        pass
        # print("\n Could not warp annotations!")
        # print(e)

    
    return atlasResult, annotationResult

if __name__ == "__main__":
    # Check if we have the nrrd files
    nrrdPath = Path.home() / ".belljar/nrrd"

    # Setup path objects
    inputPath = Path(args.input.strip())
    outputPath = Path(args.output.strip())
    # Get the file paths
    fileList = [name for name in os.listdir(inputPath) if os.path.isfile(inputPath / name) and not name.startswith('.') and name.endswith('.png')]
    absolutePaths = [str(inputPath / p) for p in fileList] 
    
    if not nrrdPath.exists():
        # If we don't have what we need, we should grab it from Allen
        os.mkdir(nrrdPath)
        # Update the load bar that we have extra tasks
        print(24, flush=True)
        print("Downloading refrence atlas...", flush=True)
        nissl = downloadFile(nisslDownloadLink, nrrdPath)
        annotation = downloadFile(annotationDownloadLink, nrrdPath)
        print("Rotating refrence atlas... this will take about an hour or more!", flush=True)
        buildRotatedAtlases(nrrdPath / nissl, nrrdPath / annotation, nrrdPath)
    else:
        # otherwise we are just processing these files, 3 steps
        print(3, flush=True)

    # Setup the images for analysis
    images = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY) for p in absolutePaths]
    resizedImages = [cv2.resize(im, (512,512)) for im in images]
    # Calculate and get the predictions
    # Predictions dict holds the section numbers for atlas
    print("Making predictions...", flush=True)
    predictions, angle = makePredictions(resizedImages, fileList, args.model.strip(), args.embeds.strip())
    # Load the appropriate atlas
    atlas, atlasHeader = nrrd.read(str(nrrdPath / f"r_nissl_{angle}.nrrd"))
    annotation, annotationHeader = nrrd.read(str(nrrdPath / f"r_annotation_{angle}.nrrd"))
    print("Awaiting fine tuning...", flush=True)
    # Setup the viewer
    viewer = napari.Viewer()
    # Add each layer
    sectionLayer = viewer.add_image(cv2.resize(images[0], (atlas.shape[2]//2,atlas.shape[1])), name="section")
    atlasLayer = viewer.add_image(atlas[:, :, :atlas.shape[2]//2], name="atlas", opacity=0.30)
    # Set the initial slider position
    viewer.dims.set_point(0, predictions[fileList[0]])
    # Track the current section
    currentSection = 0
    # Setup  the napari contorls
    # Button callbacks
    def nextSection():
        '''Move one section forward by crawling file paths'''
        global currentSection, progressBar
        if not currentSection == len(images) - 1:
            predictions[fileList[currentSection]] = viewer.dims.current_step[0]
            currentSection += 1
            progressBar.setFormat(f"{currentSection + 1}/{len(images)}")
            progressBar.setValue(currentSection + 1)
            sectionLayer.data = cv2.resize(images[currentSection], (atlas.shape[2]//2,atlas.shape[1]))
            viewer.dims.set_point(0, predictions[fileList[currentSection]])
    
    def prevSection():
        '''Move one section backward by crawling file paths'''
        global currentSection, progressBar
        if not currentSection == 0:
            predictions[fileList[currentSection]] = viewer.dims.current_step[0]
            currentSection -= 1
            progressBar.setFormat(f"{currentSection + 1}/{len(images)}")
            progressBar.setValue(currentSection + 1)
            progressBar.setValue(currentSection)
            sectionLayer.data = cv2.resize(images[currentSection], (atlas.shape[2]//2,atlas.shape[1]))
            viewer.dims.set_point(0, predictions[fileList[currentSection]])
    
    def finishAlignment():
        '''Save our final updated prediction, perform warps, close'''
        print("Writing output...", flush=True)
        global currentSection
        predictions[fileList[currentSection]] = viewer.dims.current_step[0]
        # Write the predictions to a file
        for i in range(len(images)):
            imageName = fileList[i]
            atlasWarp, annoWarp = warpToDAPI((atlas[predictions[imageName], : , :atlas.shape[2]//2]/256).astype('uint8'), 
                                              images[i], 
                                             (annotation[predictions[imageName], : , :annotation.shape[2]//2]).astype('int32')
                                            )
            cv2.imwrite(str(outputPath / f"Atlas_{imageName.split('.')[0]}.png"), atlasWarp)
            
            with open(str(outputPath / f"Annotation_{imageName.split('.')[0]}.pkl"), "wb") as annoOut:
                pickle.dump(annoWarp, annoOut)
            

            # Prep regions for saving
            regions = {}
            nameToRegion = {}
            with open(args.structures.strip()) as structureFile:
                structureReader = csv.reader(structureFile, delimiter=",")
                
                header = next(structureReader) # skip header
                root = next(structureReader) # skip atlas root region
                # manually set root, due to weird values
                regions[997] = {"acronym":"undefined", "name":"undefined", "parent":"N/A","points":[]}
                regions[0] = {"acronym":"LIW", "name":"Lost in Warp", "parent":"N/A", "points":[]}
                nameToRegion["undefined"] = 997
                nameToRegion["Lost in Warp"] = 0
                # store all other atlas regions and their linkages
                for row in structureReader:
                    regions[int(row[0])] = {"acronym":row[3], "name":row[2], "parent":int(row[8]), "points":[]}
                    nameToRegion[row[2]] = int(row[0])
            
            # Write the atlas borders ontop of dapi image
            dapi = images[i]
            y, x = annoWarp.shape
            for i in range(x-1):
                for j in range(y-1):
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
                    area = annoWarp[j, i]
                    if not all(x == area for x in surroundingPoints) and not all(x == 0 for x in surroundingPoints):
                        try:
                            if not all(regions[x]['parent'] == regions[area]['parent'] for x in surroundingPoints):
                                # We are accounting for the padding in the rotation process here
                                # Additonally write this pixel as white
                                dapi[j-101, i-101] = 255
                        except:
                            pass
                    try:
                        if all(x == area for x in surroundingPoints):
                            regions[regions[area]['parent']]['points'].append((j, i))
                    except:
                        pass
            
            # for region in regions:
            #     if regions[region]['points'] != [] and region not in [997, 0]:
            #         m = np.mean(regions[region]['points'], axis=0).astype(np.int8)
            #         try:
            #             cv2.putText(dapi, regions[region]['acronym'], (m[0] + 200, m[1] + 200) , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            #         except Exception as e:
            #             pass

            cv2.imwrite(str(outputPath / f"Map_{imageName.split('.')[0]}.png"), dapi)

        viewer.close()
        print("Done!", flush=True)
    # Button objects
    nextButton = QPushButton('Next Section')
    backButton = QPushButton('Previous Section')
    doneButton = QPushButton('Done')
    progressBar = QProgressBar(minimum=1, maximum=len(images))
    progressBar.setFormat(f"1/{len(images)}")
    progressBar.setValue(1)
    progressBar.setAlignment(Qt.AlignCenter)
    # Link callback and objects
    nextButton.clicked.connect(nextSection)
    backButton.clicked.connect(prevSection)
    doneButton.clicked.connect(finishAlignment)
    # Add them to the dock
    viewer.window.add_dock_widget([progressBar, nextButton, backButton, doneButton], name="Bell Jar Controls", area='left')
    # Start event loop to keep viewer open
    napari.run()