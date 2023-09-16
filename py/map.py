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
parser.add_argument("-a", "--spacing", help="override predicted spacing", default=False)
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


def save_alignment(
    selected_sections_left,
    selected_sections_right,
    selected_regions,
    file_list,
    angle,
    input_dir,
    masks=None,
):
    """Writes the selected atlas section for each tissue section to a simple text file"""
    with open(Path(input_dir.strip()) / "alignment.txt", "w") as f:
        f.write(f"{angle}\n")
        for i, (s, r) in enumerate(zip(selected_sections_left, selected_regions)):
            current_file = file_list[i]
            s_r = selected_sections_right[current_file]
            f.write(f"{current_file}${s}${s_r}${r}\n")

    if masks is not None:
        with open(Path(input_dir.strip()) / "masks.pkl", "wb") as f:
            pickle.dump(masks, f)


def load_masks(input_dir):
    """Load masks if any from the input directory"""
    try:
        print("Loading prior masks...", flush=True)
        with open(Path(input_dir.strip()) / "masks.pkl", "rb") as f:
            return pickle.load(f)
    except:
        # bad file or no file, just give no masks
        print("No compatible prior masks found...", flush=True)
        return {}


def load_alignment(input_dir, file_list):
    """Load prior alignments if any from the input directory"""
    alignments = {}
    alignments_right = {}
    angle = 0
    region_selections = {f: "A" for f in file_list}
    try:
        print("Loading prior alignments...", flush=True)
        with open(Path(input_dir.strip()) / "alignment.txt", "r") as f:
            lines = f.readlines()

            angle = int(lines[0])
            for line in lines[1:]:
                file_name, section_left, section_right, region = line.split("$")
                file_name = file_name.strip()
                if file_name in file_list:
                    alignments[file_name] = int(section_left)
                    alignments_right[file_name] = int(section_right)
                    region_selections[file_name] = region.strip()

            return alignments, alignments_right, angle, region_selections
    except:
        # bad file or no file, just give no alignments
        print("No compatible prior alignments found...", flush=True)
        alignments = {}
        alignments_right = {}
        angle = 0

    return alignments, alignments_right, angle, region_selections


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

    # Sort the file paths by number
    fileList.sort()

    absolutePaths = [str(inputPath / p) for p in fileList]
    # Create a dict to track which sections have been visted
    visited = {f: False for f in fileList}
    is_linked = {f: True for f in fileList}
    # Update the user, next steps are coming
    print(4 + len(absolutePaths), flush=True)

    # Setup the images for analysis
    images = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY) for p in absolutePaths]
    resizedImages = [cv2.resize(im, (256, 256)) for im in images]

    prior_alignment, prior_right, prior_angle, region_selections = load_alignment(
        args.input, fileList
    )
    did_load_alignment = len(prior_alignment) > 0

    if len(prior_alignment) > 0:
        # load in any old values
        predictions = prior_alignment
        right_hemisphere_steps = prior_right
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
            right_hemisphere_steps[image] = new_predictions[image]

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
        right_hemisphere_steps = {f: predictions[f] for f in fileList}

    # Globals for alignment
    left_section_num = predictions[fileList[0]]
    right_section_num = right_hemisphere_steps[fileList[0]]
    atlas_masks = load_masks(args.input)

    def adjustPredictions(predictions, right_predictions, visited, fileList):
        global args

        x_visited = [i for i, v in enumerate(visited.values()) if v]
        y_visited = [predictions[fileList[i]] for i in x_visited]
        y_r_visited = [right_predictions[fileList[i]] for i in x_visited]

        # Check if args.spacing exists and attempt to convert it to an integer
        spacing = None
        if hasattr(args, "spacing"):
            try:
                spacing = int(args.spacing) // 10
            except ValueError:
                spacing = 0

        # If there are not enough visited points to establish a trend,
        # we can't adjust predictions and just return the inputs unchanged
        if len(x_visited) < 2:
            return predictions, right_predictions

        # Calculate the line of best fit based on the visited points
        m, b = np.polyfit(x_visited, y_visited, 1)
        m_r, b_r = np.polyfit(x_visited, y_r_visited, 1)

        # Update the prediction values for unvisited sections based on the trend defined by the line of best fit
        for i in range(len(fileList)):
            if not visited[fileList[i]]:
                predictions[fileList[i]] = int(m * (i + spacing) + b)
                right_predictions[fileList[i]] = int(m_r * (i + spacing) + b_r)

        return predictions, right_predictions

    # Load the appropriate atlas
    atlas, atlasHeader = nrrd.read(
        str(nrrdPath / f"ara_nissl_10_all.nrrd"), index_order="C"
    )

    print("Awaiting fine tuning...", flush=True)
    # Setup the viewer
    viewer = napari.Viewer(
        title="Bell Jar Atlas Alignment",
    )
    # Add each layer

    if not did_load_alignment:
        predictions, right_hemisphere_steps = adjustPredictions(
            predictions, right_hemisphere_steps, visited, fileList
        )

    contrast_limits = [0, images[0].max()]

    if is_whole:
        left_atlas = atlas[predictions[fileList[0]], :, :]
        right_atlas = atlas[right_hemisphere_steps[fileList[0]], :, ::-1]

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
    def set_left_section():
        """Set the left hemisphere section"""
        global left_section_num, isProcessing, currentSection, predictions, right_hemisphere_steps, right_section_num
        if isProcessing:
            return

        new_left_section_num = left_hemi_slider.value()
        left_atlas_layer.data = atlas[new_left_section_num, :, :]
        left_hemi_value.setText(str(new_left_section_num))

        if link_hemispheres.isChecked() and is_whole:
            difference = new_left_section_num - left_section_num
            right_value = min(max(right_hemi_slider.value() + difference, 0), 1319)
            right_section_num = right_value
            right_hemi_slider.blockSignals(True)
            right_hemi_slider.setValue(right_value)
            right_hemi_slider.blockSignals(False)

            right_hemi_value.setText(str(right_value))
            right_hemisphere_steps[fileList[currentSection]] = right_value

            right_atlas_layer.data = atlas[right_value, :, ::-1]
        left_section_num = (
            new_left_section_num  # Update the global variable after processing
        )
        predictions[fileList[currentSection]] = new_left_section_num

    def set_right_section():
        """Set the right hemisphere section"""
        global right_section_num, isProcessing, currentSection, predictions, right_hemisphere_steps, left_section_num
        if isProcessing:
            return

        new_right_section_num = right_hemi_slider.value()
        right_atlas_layer.data = atlas[new_right_section_num, :, ::-1]
        right_hemi_value.setText(str(new_right_section_num))

        if link_hemispheres.isChecked():
            difference = new_right_section_num - right_section_num
            left_value = min(max(left_hemi_slider.value() + difference, 0), 1319)
            left_section_num = left_value
            left_hemi_slider.blockSignals(True)
            left_hemi_slider.setValue(left_value)
            left_hemi_slider.blockSignals(False)

            left_hemi_value.setText(str(left_value))
            predictions[fileList[currentSection]] = left_value

            left_atlas_layer.data = atlas[left_value, :, :]
        right_section_num = (
            new_right_section_num  # Update the global variable after processing
        )
        right_hemisphere_steps[fileList[currentSection]] = new_right_section_num

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

    def get_current_atlas():
        global left_section_num, right_section_num, is_whole, atlas
        if is_whole:
            return atlas[left_section_num, :, :], atlas[right_section_num, :, ::-1]
        else:
            return atlas[left_section_num, :, :], None

    def set_mask():
        global atlas_masks, currentSection, addMask

        left_atlas, right_atlas = get_current_atlas()
        if right_atlas is not None:
            mask = paint_and_get_mask(np.concatenate((left_atlas, right_atlas), axis=1))
        else:
            mask = paint_and_get_mask(left_atlas)

        atlas_masks[fileList[currentSection]] = mask
        addMask.setText("Change Mask")

    def paint_and_get_mask(img):
        drawing = False
        pts = []

        # ensure image is color
        if len(img.shape) == 2:
            # convert to 8bit from float64
            img = (img).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        def draw_circle(event, x, y, flags, param):
            nonlocal drawing, pts
            if event == cv2.EVENT_LBUTTONDOWN:
                drawing = True
                pts.append((x, y))
            elif event == cv2.EVENT_MOUSEMOVE:
                if drawing:
                    cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                    pts.append((x, y))
            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                pts.append((x, y))

        # Load image
        if img is None:
            print(f"Could not open or find the image")
            return None

        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        cv2.namedWindow("Click and hold to outline | Press Q to finish")
        cv2.setMouseCallback(
            "Click and hold to outline | Press Q to finish", draw_circle
        )

        while 1:
            cv2.imshow("Click and hold to outline | Press Q to finish", img)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break

        # Filling the closed shape
        if len(pts) > 2:
            cv2.fillPoly(mask, [np.array(pts)], 1)

        mask = np.logical_not(mask).astype(np.uint8)

        cv2.destroyAllWindows()
        return mask

    def nextSection():
        """Move one section forward by crawling file paths"""
        global currentSection, progressBar, isProcessing, predictions, right_hemisphere_steps, left_section_num, right_section_num
        if not currentSection == len(images) - 1:
            visited[fileList[currentSection]] = True
            is_linked[fileList[currentSection]] = link_hemispheres.isChecked()
            if not did_load_alignment:
                predictions, right_hemisphere_steps = adjustPredictions(
                    predictions, right_hemisphere_steps, visited, fileList
                )

            currentSection += 1
            progressBar.setFormat(f"{currentSection + 1}/{len(images)}")
            progressBar.setValue(currentSection + 1)
            left_section_num = predictions[fileList[currentSection]]
            right_section_num = right_hemisphere_steps[fileList[currentSection]]

            # If there is a mask set the addMask button text to Change Mask
            if fileList[currentSection] in list(atlas_masks.keys()):
                addMask.setText("Change Mask")
            else:
                addMask.setText("Add Mask")

            sectionLayer.data = cv2.resize(
                images[currentSection],
                (atlas.shape[2], atlas.shape[1]),
            )
            link_hemispheres.setChecked(is_linked[fileList[currentSection]])
            right_hemi_slider.blockSignals(True)
            right_hemi_slider.setValue(right_section_num)
            right_hemi_slider.blockSignals(False)
            left_hemi_slider.setValue(left_section_num)
            left_hemi_value.setText(str(left_section_num))
            right_hemi_value.setText(str(right_section_num))
            update_region()

    def prevSection():
        """Move one section backward by crawling file paths"""
        global currentSection, progressBar, isProcessing, predictions, right_hemisphere_steps, left_section_num, right_section_num
        if not currentSection == 0:
            visited[fileList[currentSection]] = True
            is_linked[fileList[currentSection]] = link_hemispheres.isChecked()

            if not did_load_alignment:
                predictions, right_hemisphere_steps = adjustPredictions(
                    predictions, right_hemisphere_steps, visited, fileList
                )

            currentSection -= 1
            progressBar.setFormat(f"{currentSection + 1}/{len(images)}")
            progressBar.setValue(currentSection + 1)
            progressBar.setValue(currentSection)
            left_section_num = predictions[fileList[currentSection]]
            right_section_num = right_hemisphere_steps[fileList[currentSection]]
            sectionLayer.data = cv2.resize(
                images[currentSection],
                (atlas.shape[2], atlas.shape[1]),
            )

            # If there is a mask set the addMask button text to Change Mask
            if fileList[currentSection] in list(atlas_masks.keys()):
                addMask.setText("Change Mask")
            else:
                addMask.setText("Add Mask")

            link_hemispheres.setChecked(is_linked[fileList[currentSection]])
            right_hemi_slider.blockSignals(True)
            right_hemi_slider.setValue(right_section_num)
            right_hemi_slider.blockSignals(False)
            left_hemi_slider.setValue(left_section_num)
            left_hemi_value.setText(str(left_section_num))
            right_hemi_value.setText(str(right_section_num))
            update_region()

    def finishAlignment():
        """Save our final updated prediction, perform warps, close, also write atlas borders to file"""
        global currentSection, isProcessing, angle, predictions, region_selections, atlas_masks

        if isProcessing:
            return

        print("Warping output...", flush=True)
        isProcessing = True

        # Save the seleceted sections
        save_alignment(
            list(predictions.values()),
            right_hemisphere_steps,
            list(region_selections.values()),
            fileList,
            angle,
            args.input,
            masks=atlas_masks,
        )

        # Load the appropriate annotation
        annotation, annotationHeader = nrrd.read(
            str(nrrdPath / f"annotation_10_all.nrrd"),
            index_order="C",
        )

        if "C" in region_selections.values():
            c_annotation, c_annotationHeader = nrrd.read(
                str(nrrdPath / f"annotation_10_cerebrum.nrrd"),
                index_order="C",
            )
            c_atlas, c_atlasHeader = nrrd.read(
                str(nrrdPath / f"ara_nissl_10_cerebrum.nrrd"), index_order="C"
            )

        if "NC" in region_selections.values():
            nc_annotation, nc_annotationHeader = nrrd.read(
                str(nrrdPath / f"annotation_10_no_cerebrum.nrrd"),
                index_order="C",
            )

            nc_atlas, nc_atlasHeader = nrrd.read(
                str(nrrdPath / f"ara_nissl_10_no_cerebrum.nrrd"), index_order="C"
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

            # Check if tissue has a mask
            if imageName in atlas_masks.keys():
                mask = atlas_masks[imageName]
                section = section * mask
                label = label * mask

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
            # contrast adjust the tissue
            tissue = (255 * (tissue / tissue.max())).astype(np.uint8)
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

    # Button objects
    nextButton = QPushButton("Next Section")
    backButton = QPushButton("Previous Section")
    doneButton = QPushButton("Done")
    addMask = QPushButton("Add Mask")

    progressBar = QProgressBar(minimum=1, maximum=len(images))
    progressBar.setFormat(f"1/{len(images)}")
    progressBar.setValue(1)
    progressBar.setAlignment(Qt.AlignCenter)
    # Link callback and objects
    nextButton.clicked.connect(nextSection)
    backButton.clicked.connect(prevSection)
    doneButton.clicked.connect(finishAlignment)
    addMask.clicked.connect(set_mask)
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
    widgets = [
        progressBar,
        atlasTypeDropdown,
        addMask,
        nextButton,
        backButton,
        doneButton,
    ]
    # Link left and right hemispheres
    link_hemispheres = QCheckBox("Link Hemispheres")
    link_hemispheres.setChecked(is_linked[fileList[currentSection]])
    if is_whole:
        widgets.insert(1, link_hemispheres)
        extras = [right_hemi_label, right_hemi_slider, right_hemi_value]
        bottomRow.extend(extras)

    # check if first section has loaded mask
    if fileList[currentSection] in list(atlas_masks.keys()):
        addMask.setText("Change Mask")

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
    viewer.show()
    napari.run()
