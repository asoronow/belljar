import os
import numpy as np
import cv2
import pickle
from pathlib import Path
from demons import register_to_atlas, match_histograms
from ae_tools import make_predictions
import nrrd
import SimpleITK as sitk
import csv
import napari
import argparse
from skimage.metrics import structural_similarity as ssim
from qtpy.QtWidgets import QPushButton, QProgressBar, QLabel, QCheckBox
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
    # Update the user, next steps are coming
    print(4 + len(absolutePaths), flush=True)

    # Setup the images for analysis
    images = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY) for p in absolutePaths]
    resizedImages = [cv2.resize(im, (256, 256)) for im in images]
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
    # Create a dict to track which sections have been visted
    visited = {}
    for f in fileList:
        visited[f] = False

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

        if len(x_visited) < 2:
            # compute line of best fit with all points
            x_visited = [i for i in range(len(fileList))]
            y_visited = [predictions[fileList[i]] for i in x_visited]

        m, c = best_fit_line(x_visited, y_visited)

        for i in range(len(fileList)):
            if not visited[fileList[i]]:
                predictions[fileList[i]] = int(m * i + c)

        return predictions

    # Load the appropriate atlas
    # Override the angle if needed
    atlas, atlasHeader = nrrd.read(
        str(nrrdPath / f"ara_nissl_10_all.nrrd"), index_order="C"
    )
    annotation, annotationHeader = nrrd.read(
        str(nrrdPath / f"annotation_10_all.nrrd"),
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
        left_atlas = atlas.copy()
        right_atlas = atlas.copy()[:, :, ::-1]
        # expand the dim of the right atlas
        # swap the 1st and 0th axis

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
        viewer.grid.shape = (2, 3)

    else:
        atlasLayer = viewer.add_image(
            atlas,
            name="atlas",
        )

        sectionLayer = viewer.add_image(
            cv2.resize(images[0], (atlas.shape[2], atlas.shape[1])),
            name="section",
            colormap="cyan",
            contrast_limits=contrast_limits,
        )

    # Set the initial slider position
    viewer.dims.set_point(0, predictions[fileList[0]])

    # Track the current section
    currentSection = 0
    isProcessing = False
    viewer.grid.enabled = True

    # Setup  the napari contorls
    # Button callbacks

    def nextSection():
        """Move one section forward by crawling file paths"""
        global currentSection, progressBar
        if not currentSection == len(images) - 1:
            predictions[fileList[currentSection]] = viewer.dims.current_step[0]
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

            viewer.dims.set_point(0, predictions[fileList[currentSection]])

    def prevSection():
        """Move one section backward by crawling file paths"""
        global currentSection, progressBar
        if not currentSection == 0:
            predictions[fileList[currentSection]] = viewer.dims.current_step[0]
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

            viewer.dims.set_point(0, predictions[fileList[currentSection]])

    def finishAlignment():
        """Save our final updated prediction, perform warps, close, also write atlas borders to file"""
        global currentSection, isProcessing, angle, predictions
        if isProcessing:
            return
        print("Warping output...", flush=True)
        isProcessing = True

        # Get the final section details wherever we stopped
        predictions[fileList[currentSection]] = viewer.dims.current_step[0]

        # Save the seleceted sections
        save_alignment(list(predictions.values()), fileList, angle, args.input)

        # Warp the predictions on the tissue and save the results
        for i in range(len(images)):
            imageName = fileList[i]
            print(f"Warping {imageName}...", flush=True)
            if is_whole:
                left_label = annotation[int(predictions[imageName]), :, :]
                left_section = atlas[int(predictions[imageName]), :, :]

                right_label = annotation[int(predictions[imageName]), :, ::-1]
                right_section = atlas[int(predictions[imageName]), :, ::-1]

                section = np.zeros((left_section.shape[0], left_section.shape[1] * 2))
                section[:, : left_section.shape[1]] = left_section
                section[:, left_section.shape[1] :] = right_section

                label = np.zeros(
                    (left_label.shape[0], left_label.shape[1] * 2), dtype=np.uint32
                )
                label[:, : left_label.shape[1]] = left_label
                label[:, left_label.shape[1] :] = right_label
            else:
                label = annotation[int(predictions[imageName]), :, :]
                section = atlas[int(predictions[imageName]), :, :]

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

    def predict_alignment_accuracy():
        """Uses SSIM to get a metric of alignment accuracy in realtime"""
        global currentSection
        if isProcessing:
            return

        # Get the current section
        # Get the current image
        section_image = images[currentSection]
        atlas_pos = viewer.dims.current_step[0]

        if is_whole:
            left_atlas = atlas[atlas_pos, :, :]
            right_atlas = atlas[atlas_pos, :, ::-1]

            atlas_image = np.zeros((left_atlas.shape[0], left_atlas.shape[1] * 2))
            atlas_image[:, : left_atlas.shape[1]] = left_atlas
            atlas_image[:, left_atlas.shape[1] :] = right_atlas
        else:
            atlas_image = atlas[atlas_pos, :, :]

        atlas_image = cv2.normalize(
            atlas_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
        )
        atlas_image = cv2.resize(
            atlas_image, (section_image.shape[1], section_image.shape[0])
        )

        # histograms
        section_image = sitk.GetImageFromArray(section_image)
        atlas_image = sitk.GetImageFromArray(atlas_image)

        atlas_image = match_histograms(atlas_image, section_image)

        atlas_image = sitk.GetArrayFromImage(atlas_image)
        section_image = sitk.GetArrayFromImage(section_image)

        # Get percent similarity
        result = ssim(section_image, atlas_image)

        # Update the label
        acc_label.setText(f"Structural Similarity: {100*result:.2f}%")
        # Make the label red if the accuracy is below 80%
        if result < 0.25:
            acc_label.setStyleSheet("font: 12px; color: red")
        elif result < 0.5:
            acc_label.setStyleSheet("font: 12px; color: yellow")
        else:
            acc_label.setStyleSheet("font: 12px; color: green")

    def link_hemispheres():
        """Link/Unlink hemispheres during whole brain alignment"""
        pass

    # Labels
    acc_label = QLabel("Alignment Accuracy: ")
    acc_label.setAlignment(Qt.AlignLeft)
    acc_label.setStyleSheet("font: 12px;")

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

    widgets = [progressBar, nextButton, backButton, doneButton]
    if is_whole:
        # Link left and right hemispheres
        is_linked = QCheckBox("Link Hemispheres")
        is_linked.setChecked(True)
        is_linked.stateChanged.connect(link_hemispheres)
        widgets.append(is_linked)
    viewer.window.add_dock_widget(
        acc_label,
        name="metrics",
        area="left",
    )

    # Add them to the dock
    viewer.window.add_dock_widget(
        widgets,
        name="controls",
        area="left",
    )
    # Start event loop to keep viewer open
    napari.run()
