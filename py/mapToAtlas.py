import os
import numpy as np
import cv2
import pickle
from pathlib import Path
from demons import register_to_atlas
from trainAE import makePredictions
import nrrd
import csv
import napari
import argparse
import SimpleITK as sitk
from scipy.spatial import distance as dist
from qtpy.QtWidgets import QPushButton, QProgressBar, QCheckBox
from qtpy.QtCore import Qt


parser = argparse.ArgumentParser(description="Map sections to atlas space")
parser.add_argument(
    "-o", "--output", help="output directory, only use if graphical false", default=""
)
parser.add_argument(
    "-i", "--input", help="input directory, only use if graphical false", default=""
)
parser.add_argument("-m", "--model", default="../models/predictor_encoder.pt")
parser.add_argument("-e", "--embeds", default="atlasEmbeddings.pkl")
parser.add_argument("-n", "--nrrd", help="path to nrrd files", default="")
parser.add_argument("-w", "--whole", default=False)
parser.add_argument("-a", "--angle", help="override predicted angle", default=False)
parser.add_argument(
    "-s",
    "--structures",
    help="structures file",
    default="../csv/structure_tree_safe_2017.csv",
)
parser.add_argument("-c", "--map", help="map file", default="../csv/class_map.pkl")
args = parser.parse_args()


def get_max_contour(image):
    """Apply a gaussian blur and otsu threshold to the image, then find the largest contour"""
    # Check if image is 8 bit
    if image.dtype != np.uint8:
        image = (image / 256).astype(np.uint8)

    blurry = cv2.GaussianBlur(image, (11, 11), 0)
    _, binary = cv2.threshold(blurry, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest_contour = max(contours, key=cv2.contourArea)

    return largest_contour


def get_affine_transform(src_bbox, dst_bbox):
    # Get the center and dimensions of the src and dst bounding rectangles
    src_center_x, src_center_y, src_width, src_height = src_bbox
    dst_center_x, dst_center_y, dst_width, dst_height = dst_bbox

    # Calculate the scaling factors along both axes
    scale_x = src_width / dst_width
    scale_y = src_height / dst_height

    # Calculate the translation vector
    translate_x = src_center_x - (dst_center_x * scale_x)
    translate_y = src_center_y - (dst_center_y * scale_y)

    # Construct the affine transformation matrix
    transform_matrix = np.array(
        [[scale_x, 0, translate_x + 0.05], [0, scale_y, translate_y - 0.05]]
    )

    return transform_matrix


def get_transformed_image(tissue_contour, atlas_contour, atlas_image, atlas_labels):
    # Compute the minimum bounding rectangles for the tissue and atlas contours
    tissue_rect = cv2.boundingRect(tissue_contour)
    atlas_rect = cv2.boundingRect(atlas_contour)

    # Calculate the affine transform matrix
    transform_matrix = get_affine_transform(tissue_rect, atlas_rect)

    # Apply the affine transform to the atlas image
    transformed_atlas_image = cv2.warpAffine(
        atlas_image, transform_matrix, (atlas_image.shape[1], atlas_image.shape[0])
    )

    # Apply the affine transform to the atlas labels
    transformed_atlas_labels = cv2.warpAffine(
        atlas_labels,
        transform_matrix,
        (atlas_labels.shape[1], atlas_labels.shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return transformed_atlas_image, transformed_atlas_labels.astype(np.int32)


if __name__ == "__main__":
    # Check if we have the nrrd files
    nrrdPath = Path(args.nrrd.strip())

    # Set if we are using whole or half the brain
    selectionModifier = 2 if not eval(args.whole) else 1

    # Setup path objects
    inputPath = Path(args.input.strip())
    outputPath = Path(args.output.strip())
    # Get the file paths
    fileList = [
        name
        for name in os.listdir(inputPath)
        if os.path.isfile(inputPath / name)
        and not name.startswith(".")
        and name.endswith(".png")
    ]
    absolutePaths = [str(inputPath / p) for p in fileList]

    # Update the user, next steps are coming
    print(3, flush=True)

    # Setup the images for analysis
    images = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY) for p in absolutePaths]
    resizedImages = [cv2.resize(im, (512, 512)) for im in images]
    # Calculate and get the predictions
    # Predictions dict holds the section numbers for atlas
    print("Making predictions...", flush=True)
    predictions, angle, normalizedImages = makePredictions(
        resizedImages,
        fileList,
        args.model.strip(),
        args.embeds.strip(),
        hemisphere=eval(args.whole),
    )

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
                    averageIncrease += (
                        predictions[fileList[i]] - predictions[fileList[i - 1]]
                    )
            averageIncrease /= sum(visited.values())
            # Adjust the predictions of the unvisted sections
            for i in range(len(fileList)):
                if not visited[fileList[i]]:
                    predictions[fileList[i]] = (
                        predictions[fileList[i - 1]] + averageIncrease
                    )

        return predictions

    # Load the appropriate atlas
    # Override the angle if needed
    angle = int(args.angle.strip()) if not int(args.angle.strip()) == 99 else angle
    atlas, atlasHeader = nrrd.read(str(nrrdPath / f"r_nissl_{angle}.nrrd"))
    annotation, annotationHeader = nrrd.read(
        str(nrrdPath / f"r_annotation_{angle}.nrrd")
    )
    print("Awaiting fine tuning...", flush=True)
    # Setup the viewer
    viewer = napari.Viewer()
    # Add each layer
    sectionLayer = viewer.add_image(
        cv2.resize(
            normalizedImages[0], (atlas.shape[2] // selectionModifier, atlas.shape[1])
        ),
        name="section",
    )
    atlasLayer = viewer.add_image(
        atlas[:, :, : atlas.shape[2] // selectionModifier], name="atlas", opacity=0.30
    )
    # Set the initial slider position
    viewer.dims.set_point(0, predictions[fileList[0]])
    # Track the current section
    currentSection = 0
    isProcessing = False
    # Setup  the napari contorls
    # Button callbacks

    def nextSection():
        """Move one section forward by crawling file paths"""
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
            progressBar.setFormat(f"{currentSection + 1}/{len(normalizedImages)}")
            progressBar.setValue(currentSection + 1)
            sectionLayer.data = cv2.resize(
                normalizedImages[currentSection],
                (atlas.shape[2] // selectionModifier, atlas.shape[1]),
            )
            viewer.dims.set_point(0, predictions[fileList[currentSection]])

    def prevSection():
        """Move one section backward by crawling file paths"""
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
            progressBar.setFormat(f"{currentSection + 1}/{len(normalizedImages)}")
            progressBar.setValue(currentSection + 1)
            progressBar.setValue(currentSection)
            sectionLayer.data = cv2.resize(
                normalizedImages[currentSection],
                (atlas.shape[2] // selectionModifier, atlas.shape[1]),
            )
            viewer.dims.set_point(0, predictions[fileList[currentSection]])

    def finishAlignment():
        """Save our final updated prediction, perform warps, close, also write atlas borders to file"""
        global currentSection, separatedCheckbox, isProcessing
        if isProcessing:
            return
        print("Warping output...", flush=True)
        isProcessing = True
        predictions[fileList[currentSection]] = viewer.dims.current_step[0]
        if separatedCheckbox.isChecked():
            separated[fileList[currentSection]] = True
        else:
            separated[fileList[currentSection]] = False
        # Write the predictions to a file
        for i in range(len(images)):
            imageName = fileList[i]
            # print(separated[imageName])
            if selectionModifier == 2:
                x_val = annotation.shape[2] // 2
                pre_label = annotation[int(predictions[imageName]), :, :x_val]
                pre_section = atlas[int(predictions[imageName]), :, :x_val]

                label = np.zeros((pre_label.shape[0], x_val), dtype=np.uint32)
                label[:, : pre_label.shape[1] - 50] = pre_label[:, 50:x_val]

                section = np.zeros((pre_section.shape[0], x_val))
                section[:, : pre_section.shape[1] - 50] = pre_section[:, 50:x_val]

            else:
                x_val = annotation.shape[2]
                label = annotation[int(predictions[imageName]), :, :x_val]

                section = atlas[int(predictions[imageName]), :, :x_val]

            tissue = images[i]

            # resize atlas and label to match tissue
            section = cv2.resize(section, (tissue.shape[1], tissue.shape[0]))
            label = cv2.resize(
                label.astype(np.float64),
                (tissue.shape[1], tissue.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            tissue_contour = get_max_contour(tissue)
            atlas_contour = get_max_contour(section)

            transformed_atlas_image, transformed_atlas_labels = get_transformed_image(
                tissue_contour, atlas_contour, section, label.astype(np.float64)
            )

            warped_labels, warped_atlas, color_label = register_to_atlas(
                tissue,
                transformed_atlas_image,
                transformed_atlas_labels,
                args.map.strip(),
            )

            cv2.imwrite(
                str(outputPath / f"Atlas_{imageName.split('.')[0]}.png"), warped_atlas
            )
            # write label
            cv2.imwrite(
                str(outputPath / f"Label_{imageName.split('.')[0]}.png"), color_label
            )

            with open(
                str(outputPath / f"Annotation_{imageName.split('.')[0]}.pkl"), "wb"
            ) as annoOut:
                pickle.dump(warped_labels, annoOut)

            # Prep regions for saving
            regions = {}
            nameToRegion = {}
            with open(args.structures.strip()) as structureFile:
                structureReader = csv.reader(structureFile, delimiter=",")

                header = next(structureReader)  # skip header
                root = next(structureReader)  # skip atlas root region
                # manually set root, due to weird values
                regions[997] = {
                    "acronym": "undefined",
                    "name": "undefined",
                    "parent": "N/A",
                    "points": [],
                }
                regions[0] = {
                    "acronym": "LIW",
                    "name": "Lost in Warp",
                    "parent": "N/A",
                    "points": [],
                }
                nameToRegion["undefined"] = 997
                nameToRegion["Lost in Warp"] = 0
                # store all other atlas regions and their linkages
                for row in structureReader:
                    regions[int(row[0])] = {
                        "acronym": row[3],
                        "name": row[2],
                        "parent": int(row[8]),
                        "points": [],
                    }
                    nameToRegion[row[2]] = int(row[0])
        viewer.close()

        print("Done!", flush=True)

    # Button objects
    nextButton = QPushButton("Next Section")
    backButton = QPushButton("Previous Section")
    doneButton = QPushButton("Done")
    # Add checkbox for seperated
    separatedCheckbox = QCheckBox("Seperated?")
    progressBar = QProgressBar(minimum=1, maximum=len(images))
    progressBar.setFormat(f"1/{len(images)}")
    progressBar.setValue(1)
    progressBar.setAlignment(Qt.AlignCenter)
    # Link callback and objects
    nextButton.clicked.connect(nextSection)
    backButton.clicked.connect(prevSection)
    doneButton.clicked.connect(finishAlignment)
    # Add them to the dock
    viewer.window.add_dock_widget(
        [progressBar, nextButton, backButton, separatedCheckbox, doneButton],
        name="Bell Jar Controls",
        area="left",
    )
    # Start event loop to keep viewer open
    napari.run()
