import os
import numpy as np
import cv2
import pickle
from pathlib import Path
from demons import register_to_atlas
from ae_tools import make_predictions
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


def get_max_contour(image, separated=False):
    """Apply a gaussian blur and otsu threshold to the image, then find the largest contour"""
    # Check if image is 8 bit
    if image.dtype != np.uint8:
        image = (image / 256).astype(np.uint8)

    blurry = cv2.GaussianBlur(image, (11, 11), 0)
    _, binary = cv2.threshold(blurry, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if separated:
        # if tissue is far apart we should take the two largest and combine them
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        largest_contour = np.concatenate((sorted_contours[0], sorted_contours[1]))
    else:
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
    transform_matrix = np.array([[scale_x, 0, translate_x], [0, scale_y, translate_y]])

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


def save_alignment(selected_sections, file_list, angle, input_dir):
    """Writes the selected atlas section for each tissue section to a simple text file"""
    with open(Path(input_dir.strip()) / "alignment.txt", "w") as f:
        f.write(f"{angle}\n")
        for i, s in enumerate(selected_sections):
            f.write(f"{file_list[i]}-{s}\n")


def load_alignment(input_dir, file_list):
    """Load prior alignments if any from the input directory"""
    alignments = {}
    angle = 0
    try:
        print("Loading prior alignments...", flush=True)
        with open(Path(input_dir.strip()) / "alignment.txt", "r") as f:
            lines = f.readlines()

            angle = int(lines[0])
            for line in lines[1:]:
                file_name, section = line.split("-")
                section = section.strip()
                file_name = file_name.strip()
                if file_name in file_list:
                    alignments[file_name] = int(section)

            return alignments, angle
    except:
        # bad file or no file, just give no alignments
        print("No prior alignments found...", flush=True)
        alignments = {}
        angle = 0
    finally:
        return alignments, angle


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
    print(4 + len(absolutePaths), flush=True)

    # Setup the images for analysis
    images = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY) for p in absolutePaths]
    resizedImages = [cv2.resize(im, (512, 512)) for im in images]
    # Calculate and get the predictions
    # Predictions dict holds the section numbers for atlas

    prior_alignment, prior_angle = load_alignment(args.input, fileList)
    did_load_alignment = len(prior_alignment) > 0

    if len(prior_alignment) > 0:
        # load in any old values
        predictions = prior_alignment
        angle = prior_angle

        # get any new sections
        new_sections = [x for x in fileList if x not in prior_alignment.keys()]
        if len(new_sections) > 0:
            print("Making predictions for new sections...", flush=True)
        new_images = [
            cv2.cvtColor(cv2.imread(os.path.join(inputPath, p)), cv2.COLOR_BGR2GRAY)
            for p in new_sections
        ]
        new_resized_images = [cv2.resize(im, (512, 512)) for im in new_images]

        new_predictions, _ = make_predictions(
            new_resized_images,
            new_sections,
            args.model.strip(),
            args.embeds.strip(),
            hemisphere=eval(args.whole),
        )

        for i, image in enumerate(new_sections):
            predictions[image] = new_predictions[image]

    else:
        print("Making predictions...", flush=True)
        # no saved values, make a fresh prediction
        predictions, angle = make_predictions(
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
                    predictions[fileList[i]] = predictions[fileList[i - 1]] + int(
                        averageIncrease
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

    if images[0].dtype == np.uint8:
        contrast_limits = [0, images[0].max()]
    elif images[0].dtype == np.uint16:
        contrast_limits = [0, images[0].max()]

    sectionLayer = viewer.add_image(
        cv2.resize(images[0], (atlas.shape[2] // selectionModifier, atlas.shape[1])),
        name="section",
        colormap="cyan",
        contrast_limits=contrast_limits,
    )
    atlasLayer = viewer.add_image(
        atlas[:, :, : atlas.shape[2] // selectionModifier],
        name="atlas",
    )
    # Set the initial slider position
    viewer.dims.set_point(0, predictions[fileList[0]])
    viewer.grid.enabled = True
    # Track the current section
    currentSection = 0
    isProcessing = False
    # Setup  the napari contorls
    # Button callbacks

    def nextSection():
        """Move one section forward by crawling file paths"""
        global currentSection, progressBar, separatedCheckbox
        if not currentSection == len(images) - 1:
            predictions[fileList[currentSection]] = viewer.dims.current_step[0]
            visited[fileList[currentSection]] = True

            if separatedCheckbox.isChecked():
                separated[fileList[currentSection]] = True
            else:
                separated[fileList[currentSection]] = False

            if not did_load_alignment:
                adjustPredictions(predictions, visited, fileList)
            currentSection += 1
            if separated[fileList[currentSection]]:
                separatedCheckbox.setChecked(True)
            else:
                separatedCheckbox.setChecked(False)

            progressBar.setFormat(f"{currentSection + 1}/{len(images)}")
            progressBar.setValue(currentSection + 1)
            sectionLayer.data = cv2.resize(
                images[currentSection],
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
            if not did_load_alignment:
                adjustPredictions(predictions, visited, fileList)
            currentSection -= 1
            if separated[fileList[currentSection]]:
                separatedCheckbox.setChecked(True)
            else:
                separatedCheckbox.setChecked(False)
            progressBar.setFormat(f"{currentSection + 1}/{len(images)}")
            progressBar.setValue(currentSection + 1)
            progressBar.setValue(currentSection)
            sectionLayer.data = cv2.resize(
                images[currentSection],
                (atlas.shape[2] // selectionModifier, atlas.shape[1]),
            )
            viewer.dims.set_point(0, predictions[fileList[currentSection]])

    def finishAlignment():
        """Save our final updated prediction, perform warps, close, also write atlas borders to file"""
        global currentSection, separatedCheckbox, isProcessing, angle
        if isProcessing:
            return
        print("Warping output...", flush=True)
        isProcessing = True

        # Get the final section details wherever we stopped
        predictions[fileList[currentSection]] = viewer.dims.current_step[0]
        if separatedCheckbox.isChecked():
            separated[fileList[currentSection]] = True
        else:
            separated[fileList[currentSection]] = False

        # Save the seleceted sections
        save_alignment(list(predictions.values()), fileList, angle, args.input)

        # Warp the predictions on the tissue and save the results
        for i in range(len(images)):
            imageName = fileList[i]
            print(f"Warping {imageName}...", flush=True)
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
            warped_labels, warped_atlas, color_label = register_to_atlas(
                tissue,
                section,
                label.astype(np.uint32),
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
