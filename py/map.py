import os
import numpy as np
import cv2
import pickle
from pathlib import Path
from demons import register_to_atlas
from ae_tools import make_predictions
import nrrd
import SimpleITK as sitk
import csv
import napari
import argparse
from qtpy.QtWidgets import (
    QPushButton,
    QProgressBar,
    QLabel,
    QCheckBox,
    QComboBox,
    QSlider,
)
from qtpy.QtCore import Qt, QTimer


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


def save_alignment(selected_sections, selected_regions, file_list, angle, input_dir):
    """Writes the selected atlas section for each tissue section to a simple text file"""
    with open(Path(input_dir.strip()) / "alignment.txt", "w") as f:
        f.write(f"{angle}\n")
        for i, (s, r) in enumerate(zip(selected_sections, selected_regions)):
            f.write(f"{file_list[i]}-{s}-{r}\n")


def load_alignment(input_dir, file_list):
    """Load prior alignments if any from the input directory"""
    alignments = {}
    angle = 0
    region_selections = {f: "A" for f in file_list}
    try:
        print("Loading prior alignments...", flush=True)
        with open(Path(input_dir.strip()) / "alignment.txt", "r") as f:
            lines = f.readlines()

            angle = int(lines[0])
            for line in lines[1:]:
                file_name, section, region = line.split("-")
                section = section.strip()
                file_name = file_name.strip()
                if file_name in file_list:
                    alignments[file_name] = int(section)
                    region_selections[file_name] = region.strip()

            return alignments, angle, region_selections
    except:
        # bad file or no file, just give no alignments
        print("No prior alignments found...", flush=True)
        alignments = {}
        angle = 0

    return alignments, angle, region_selections


if __name__ == "__main__":
    # Check if we have the nrrd files
    nrrdPath = Path(args.nrrd.strip())

    # Set if we are using whole or half the brain
    is_whole = eval(args.whole)

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
    # Create a dict to track which sections have been visted
    visited = {f: False for f in fileList}
    # Track any right hemisphere steps
    right_hemisphere_steps = {}
    # Track any region selections
    # Update the user, next steps are coming
    print(4 + len(absolutePaths), flush=True)

    # Setup the images for analysis
    images = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY) for p in absolutePaths]
    resizedImages = [cv2.resize(im, (256, 256)) for im in images]

    prior_alignment, prior_angle, region_selections = load_alignment(
        args.input, fileList
    )
    did_load_alignment = len(prior_alignment) > 0

    left_section_num = 0
    right_section_num = 0
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
        new_resized_images = [cv2.resize(im, (256, 256)) for im in new_images]

        new_predictions = make_predictions(
            new_resized_images,
            new_sections,
            args.model.strip(),
            args.embeds.strip(),
            nrrdPath,
            hemisphere=eval(args.whole),
        )

        for i, image in enumerate(new_sections):
            predictions[image] = new_predictions[image]

    else:
        print("Making predictions...", flush=True)
        # no saved values, make a fresh prediction
        angle = 0
        predictions = make_predictions(
            resizedImages,
            fileList,
            args.model.strip(),
            args.embeds.strip(),
            nrrdPath,
            hemisphere=eval(args.whole),
        )

    def best_fit_line(x, y):
        n = len(x)

        denominator = n * sum(i**2 for i in x) - sum(x) ** 2

        # Ensure we're not dividing by zero
        if denominator == 0:
            return None, None

        m = (n * sum(i * j for i, j in zip(x, y)) - sum(x) * sum(y)) / denominator
        c = (sum(y) - m * sum(x)) / n
        return m, c

    def adjustPredictions(predictions, visited, fileList):
        x_visited = [i for i, v in enumerate(visited.values()) if v]
        y_visited = [predictions[fileList[i]] for i in x_visited]

        if len(x_visited) < 2 and len(predictions) > 1:
            # compute line of best fit with all points
            x_visited = [i for i in range(len(fileList))]
            y_visited = [predictions[fileList[i]] for i in x_visited]
        else:
            return predictions

        m, c = best_fit_line(x_visited, y_visited)

        for i in range(len(fileList)):
            if not visited[fileList[i]]:
                predictions[fileList[i]] = int(m * i + c)

        return predictions

    # Load the appropriate atlas
    atlas, atlasHeader = nrrd.read(
        str(nrrdPath / f"ara_nissl_10_all.nrrd"), index_order="C"
    )
    c_atlas, c_atlasHeader = nrrd.read(
        str(nrrdPath / f"ara_nissl_10_cerebrum.nrrd"), index_order="C"
    )
    nc_atlas, nc_atlasHeader = nrrd.read(
        str(nrrdPath / f"ara_nissl_10_no_cerebrum.nrrd"), index_order="C"
    )

    # Load the appropriate annotation
    annotation, annotationHeader = nrrd.read(
        str(nrrdPath / f"annotation_10_all.nrrd"),
        index_order="C",
    )
    c_annotation, c_annotationHeader = nrrd.read(
        str(nrrdPath / f"annotation_10_cerebrum.nrrd"),
        index_order="C",
    )
    nc_annotation, nc_annotationHeader = nrrd.read(
        str(nrrdPath / f"annotation_10_no_cerebrum.nrrd"),
        index_order="C",
    )

    print("Awaiting fine tuning...", flush=True)
    # Setup the viewer
    viewer = napari.Viewer()
    # Add each layer
    predictions = adjustPredictions(predictions, visited, fileList)

    if images[0].dtype == np.uint8:
        contrast_limits = [0, images[0].max()]
    elif images[0].dtype == np.uint16:
        contrast_limits = [0, images[0].max()]

    if is_whole:
        left_atlas = atlas[predictions[fileList[0]], :, :]
        right_atlas = atlas[predictions[fileList[0]], :, ::-1]

        right_atlas_layer = viewer.add_image(
            right_atlas,
            name="right atlas",
        )
        left_atlas_layer = viewer.add_image(
            left_atlas,
            name="left atlas",
        )
        sectionLayer = viewer.add_image(
            cv2.resize(images[0], (atlas.shape[2], atlas.shape[1])),
            name="section",
            colormap="cyan",
            contrast_limits=contrast_limits,
        )

        # set grid witdth to 3
        viewer.grid.shape = (1, 3)

    else:
        left_atlas = atlas[predictions[fileList[0]], :, :]
        left_atlas_layer = viewer.add_image(
            left_atlas,
            name="atlas",
        )

        sectionLayer = viewer.add_image(
            cv2.resize(images[0], (atlas.shape[2], atlas.shape[1])),
            name="section",
            colormap="cyan",
            contrast_limits=contrast_limits,
        )
    # Track the current section
    currentSection = 0
    isProcessing = False
    viewer.grid.enabled = True

    # Setup  the napari contorls
    # Button callbacks
    def update_step():
        global currentSection

        left_hemi_slider.setValue(predictions[fileList[currentSection]])
        left_hemi_value.setText(str(predictions[fileList[currentSection]]))
        if is_whole:
            right_hemi_slider.setValue(predictions[fileList[currentSection]])
            right_hemi_value.setText(str(predictions[fileList[currentSection]]))

    def nextSection():
        """Move one section forward by crawling file paths"""
        global currentSection, progressBar
        if not currentSection == len(images) - 1:
            visited[fileList[currentSection]] = True

            if not did_load_alignment:
                adjustPredictions(predictions, visited, fileList)

            currentSection += 1
            progressBar.setFormat(f"{currentSection + 1}/{len(images)}")
            progressBar.setValue(currentSection + 1)

            sectionLayer.data = cv2.resize(
                images[currentSection],
                (atlas.shape[2], atlas.shape[1]),
            )
            atlasTypeDropdown.setCurrentIndex(0)
            update_step()
            update_region()

    def prevSection():
        """Move one section backward by crawling file paths"""
        global currentSection, progressBar
        if not currentSection == 0:
            visited[fileList[currentSection]] = True

            if not did_load_alignment:
                adjustPredictions(predictions, visited, fileList)

            currentSection -= 1
            progressBar.setFormat(f"{currentSection + 1}/{len(images)}")
            progressBar.setValue(currentSection + 1)
            progressBar.setValue(currentSection)

            sectionLayer.data = cv2.resize(
                images[currentSection],
                (atlas.shape[2], atlas.shape[1]),
            )

            update_step()
            update_region()

    def finishAlignment():
        """Save our final updated prediction, perform warps, close, also write atlas borders to file"""
        global currentSection, isProcessing, angle, predictions, region_selections
        if isProcessing:
            return
        print("Warping output...", flush=True)
        isProcessing = True

        # Save the seleceted sections
        save_alignment(
            list(predictions.values()), region_selections, fileList, angle, args.input
        )

        # Warp the predictions on the tissue and save the results
        for i in range(len(images)):
            imageName = fileList[i]
            print(f"Warping {imageName}...", flush=True)
            used_atlas = atlas
            used_annotation = annotation
            if region_selections[imageName] == "C":
                used_atlas = c_atlas
                used_annotation = c_annotation
            elif region_selections[imageName] == "NC":
                used_atlas = nc_atlas
                used_annotation = nc_annotation

            if is_whole:
                left_label = used_annotation[int(predictions[imageName]), :, :]
                left_section = used_atlas[int(predictions[imageName]), :, :]

                if imageName in right_hemisphere_steps.keys():
                    right_section = used_atlas[
                        right_hemisphere_steps[imageName], :, ::-1
                    ]
                    right_label = used_annotation[
                        right_hemisphere_steps[imageName], :, ::-1
                    ]
                else:
                    right_label = used_annotation[int(predictions[imageName]), :, ::-1]
                    right_section = used_atlas[int(predictions[imageName]), :, ::-1]

                section = np.zeros((left_section.shape[0], left_section.shape[1] * 2))
                section[:, : left_section.shape[1]] = left_section
                section[:, left_section.shape[1] :] = right_section

                label = np.zeros(
                    (left_label.shape[0], left_label.shape[1] * 2), dtype=np.uint32
                )
                label[:, : left_label.shape[1]] = left_label
                label[:, left_label.shape[1] :] = right_label
            else:
                label = used_annotation[int(predictions[imageName]), :, :]
                section = used_atlas[int(predictions[imageName]), :, :]

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

            # composite the warped labels onto the tissue
            tissue = cv2.cvtColor(tissue, cv2.COLOR_GRAY2BGR)
            # convert color_label to 8 bit
            color_label = (color_label).astype(np.uint8)
            color_label = cv2.cvtColor(color_label, cv2.COLOR_RGB2BGR)
            tissue = cv2.addWeighted(tissue, 0.5, color_label, 0.5, 0)

            cv2.imwrite(
                str(outputPath / f"Composite_{imageName.split('.')[0]}.png"), tissue
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

    def set_left_section():
        """Set the left hemisphere section"""
        global left_section_num, isProcessing, currentSection
        if isProcessing:
            return
        left_section_num = left_hemi_slider.value()
        left_atlas_layer.data = atlas[left_section_num, :, :]
        left_hemi_value.setText(str(left_section_num))
        predictions[fileList[currentSection]] = left_section_num
        if link_hemispheres.isChecked() and is_whole:
            right_hemi_slider.setValue(left_section_num)
            right_hemi_value.setText(str(left_section_num))

    def set_right_section():
        """Set the right hemisphere section"""
        global right_section_num, isProcessing
        if isProcessing:
            return

        right_section_num = right_hemi_slider.value()
        right_atlas_layer.data = atlas[right_section_num, :, ::-1]
        right_hemi_value.setText(str(right_section_num))
        if link_hemispheres.isChecked():
            left_hemi_slider.setValue(right_section_num)
            left_hemi_value.setText(str(right_section_num))
            predictions[fileList[currentSection]] = right_section_num
        else:
            right_hemisphere_steps[fileList[currentSection]] = right_section_num

    def change_region_type():
        """Change the region type"""
        global region_selections
        current_text = atlasTypeDropdown.currentText()
        if current_text == "All Regions":
            region_selections[fileList[currentSection]] = "A"
        elif current_text == "Cerebrum Only":
            region_selections[fileList[currentSection]] = "C"
        elif current_text == "No Cerebrum":
            region_selections[fileList[currentSection]] = "NC"

    def update_region():
        global region_selections
        flag = region_selections[fileList[currentSection]]
        if flag == "A":
            atlasTypeDropdown.setCurrentIndex(0)
        elif flag == "C":
            atlasTypeDropdown.setCurrentIndex(1)
        elif flag == "NC":
            atlasTypeDropdown.setCurrentIndex(2)

    # Button objects
    nextButton = QPushButton("Next Section")
    backButton = QPushButton("Previous Section")
    doneButton = QPushButton("Done")

    progressBar = QProgressBar(minimum=1, maximum=len(images))
    progressBar.setFormat(f"1/{len(images)}")
    progressBar.setValue(1)
    progressBar.setAlignment(Qt.AlignCenter)
    # Link callback and objects
    nextButton.clicked.connect(nextSection)
    backButton.clicked.connect(prevSection)
    doneButton.clicked.connect(finishAlignment)

    # Dropdown to select if it should be cerebrum only, no cerebrum, or whole brain
    atlasTypeDropdown = QComboBox()
    atlasTypeDropdown.addItem("All Regions")
    atlasTypeDropdown.addItem("Cerebrum Only")
    atlasTypeDropdown.addItem("No Cerebrum")
    atlasTypeDropdown.currentIndexChanged.connect(change_region_type)

    # Left and right hemisphere sliders
    left_hemi_slider = QSlider(Qt.Horizontal)
    right_hemi_slider = QSlider(Qt.Horizontal)
    left_hemi_slider.setRange(0, 1319)
    right_hemi_slider.setRange(0, 1319)
    left_hemi_slider.setValue(predictions[fileList[currentSection]])
    right_hemi_slider.setValue(predictions[fileList[currentSection]])
    # add label
    left_hemi_label = QLabel("Left Hemisphere")
    right_hemi_label = QLabel("Right Hemisphere")
    left_hemi_label.setAlignment(Qt.AlignCenter)
    right_hemi_label.setAlignment(Qt.AlignCenter)
    left_hemi_value = QLabel(str(predictions[fileList[currentSection]]))
    right_hemi_value = QLabel(str(predictions[fileList[currentSection]]))

    left_hemi_slider.valueChanged.connect(set_left_section)
    right_hemi_slider.valueChanged.connect(set_right_section)

    bottomRow = [left_hemi_label, left_hemi_slider, left_hemi_value]
    widgets = [progressBar, atlasTypeDropdown, nextButton, backButton, doneButton]
    # Link left and right hemispheres
    link_hemispheres = QCheckBox("Link Hemispheres")
    link_hemispheres.setChecked(True)
    if is_whole:
        widgets.insert(1, link_hemispheres)
        extras = [right_hemi_label, right_hemi_slider, right_hemi_value]
        for e in extras:
            bottomRow.append(e)

    # Add the buttons to the dock
    viewer.window.add_dock_widget(
        bottomRow,
        name="region",
        area="bottom",
    )

    # Add them to the dock
    viewer.window.add_dock_widget(
        widgets,
        name="controls",
        area="left",
    )
    # Start event loop to keep viewer open
    napari.run()
