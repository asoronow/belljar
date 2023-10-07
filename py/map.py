import os
import numpy as np
import cv2
import pickle
from pathlib import Path
from demons import register_to_atlas
from slice_atlas import slice_3d_volume, remove_fragments
from model import TissuePredictor
import nrrd
import torch
import napari
import argparse
from qtpy.QtWidgets import (
    QPushButton,
    QProgressBar,
    QLabel,
    QComboBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)


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


class AtlasSlice:
    """
    Helper object to manage atlas slices

    Parameters:
        ap_position (int): the ap position of the slice
        x_angle (float): the x angle of the slice
        y_angle (float): the y angle of the slice
    """

    def __init__(
        self, section_name, ap_position, x_angle, y_angle, whole_brain=False, region="A"
    ):
        self.section_name = section_name
        self.ap_position = int(ap_position)
        self.x_angle = float(x_angle)
        self.y_angle = float(y_angle)
        self.region = region
        self.image = None
        self.label = None
        self.mask = None
        self.whole_brain = whole_brain

    def set_mask(self):
        drawing = False
        pts = []

        img = self.image.copy()
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

        while True:
            cv2.imshow("Click and hold to outline | Press Q to finish", img)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break

        # Filling the closed shape
        if len(pts) > 2:
            cv2.fillPoly(mask, [np.array(pts)], 1)

        mask = np.logical_not(mask).astype(np.uint8)

        cv2.destroyAllWindows()
        self.mask = mask

    def set_slice(self, atlas, annotation):
        """
        Get the slice from the atlas and annotation

        Args:
            atlas (numpy.ndarray): the atlas
            annotation (numpy.ndarray): the annotation

        Returns:
            numpy.ndarray: the atlas slice
            numpy.ndarray: the annotation slice
        """
        if self.whole_brain:
            left_image = slice_3d_volume(
                atlas, self.ap_position, self.x_angle, self.y_angle
            ).astype(np.uint8)
            right_image = slice_3d_volume(
                atlas[:, :, ::-1], self.ap_position, -1 * self.x_angle, self.y_angle
            ).astype(np.uint8)

            left_label = slice_3d_volume(
                annotation, self.ap_position, self.x_angle, self.y_angle
            ).astype(np.uint32)

            right_label = slice_3d_volume(
                annotation[:, :, ::-1],
                self.ap_position,
                -1 * self.x_angle,
                self.y_angle,
            ).astype(np.uint32)

            self.image = np.concatenate((left_image, right_image), axis=1)
            self.label = np.concatenate((left_label, right_label), axis=1)
        else:
            self.image = slice_3d_volume(
                atlas, self.ap_position, self.x_angle, self.y_angle
            ).astype(np.uint8)
            self.label = slice_3d_volume(
                annotation, self.ap_position, self.x_angle, self.y_angle
            ).astype(np.uint32)

    def get_registered(self, tissue):
        """
        Runs multi-modal registration between this atlas slice and the provided tissue section.

        Args:
            tissue (numpy.ndarray): the tissue section

        Returns:
            numpy.ndarray: the warped atlas slice
            numpy.ndarray: the warped annotation slice
            numpy.ndarray: the color annotation slice
        """
        if self.mask is not None:
            self.image = self.image * self.mask
            self.label = self.label * self.mask

        warped_labels, warped_atlas, color_label = register_to_atlas(
            tissue,
            self.image,
            self.label,
            args.map.strip(),
        )

        return warped_labels, warped_atlas, color_label


class AlignmentController:
    """
    Handles the control flow for alignment to the atlas

    Args:
        nrrd_path (str): path to nrrd files
        is_whole (bool): if we are using the whole brain or just half
        input_path (str): path to input images
        output_path (str): path to output alignments
        model_path (str): path to tissue predictor model
        spacing (int): the spacing between sections in microns
        structures_path (str): path to structures file
    """

    def __init__(
        self,
        nrrd_path,
        input_path,
        output_path,
        structures_path,
        model_path,
        spacing=None,
        is_whole=True,
    ):
        self.nrrd_path = nrrd_path
        self.input_path = input_path
        self.output_path = output_path
        self.structures_path = structures_path
        self.model_path = model_path
        self.spacing = spacing
        self.is_whole = is_whole

        self.viewer = napari.Viewer(
            title="Atlas Alignment",
        )

        self.atlas = nrrd.read(
            Path(self.nrrd_path) / "ara_nissl_10_all.nrrd", index_order="C"
        )[0]
        self.annotation = nrrd.read(
            Path(self.nrrd_path) / "annotation_10_all.nrrd", index_order="C"
        )[0]

        # Atlas layer
        self.atlas_layer = self.viewer.add_image(
            np.zeros((1920, 1080)),
            name="Atlas",
            colormap="gray",
            contrast_limits=[0, 255],
        )

        # Tissue layer
        self.tissue_layer = self.viewer.add_image(
            np.zeros((1920, 1080)),
            name="Tissue",
            colormap="gray",
            contrast_limits=[0, 255],
        )
        # make atalas 8 bit
        self.atlas = cv2.normalize(self.atlas, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        self.file_list = []
        self.num_slices = 0
        self.atlas_slices = {}

        self.visited = 0  # The index of the furthest visited section
        self.current_section = 0  # The index of the current section

        self.x_angle_spinbox = QDoubleSpinBox()
        self.x_angle_spinbox.setRange(-10, 10)
        self.x_angle_spinbox.setSingleStep(0.1)
        self.x_angle_spinbox.setValue(0)
        self.x_angle_spinbox.setSuffix("°")
        self.x_angle_spinbox.valueChanged.connect(self.update_slice)

        self.y_angle_spinbox = QDoubleSpinBox()
        self.y_angle_spinbox.setRange(-10, 10)
        self.y_angle_spinbox.setSingleStep(0.1)
        self.y_angle_spinbox.setValue(0)
        self.y_angle_spinbox.setSuffix("°")
        self.y_angle_spinbox.valueChanged.connect(self.update_slice)

        self.ap_position_spinbox = QDoubleSpinBox()
        self.ap_position_spinbox.setRange(0, 1319)
        self.ap_position_spinbox.setSingleStep(5)
        # no decimal places
        self.ap_position_spinbox.setDecimals(0)
        self.ap_position_spinbox.valueChanged.connect(self.update_position)

        # Region selection
        self.region_tags = {
            "All Regions": "A",
            "Cerebrum Only": "C",
            "No Cerebrum": "NC",
        }
        self.region_selection = QComboBox()
        self.region_selection.addItems(
            [
                "All Regions",
                "Cerebrum Only",
                "No Cerebrum",
            ]
        )
        self.region_selection.currentIndexChanged.connect(self.update_slice)

        self.mask_button = QPushButton("Set Mask")
        self.mask_button.clicked.connect(self.update_mask)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(1, self.num_slices)
        self.progress_bar.setValue(1)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_section)

        self.previous_button = QPushButton("Previous")
        self.previous_button.clicked.connect(self.previous_section)

        self.finish_button = QPushButton("Finish")
        self.finish_button.clicked.connect(self.finish)

        x_angle_widget = QWidget()
        x_angle_layout = QVBoxLayout()
        x_angle_layout.addWidget(QLabel("X Angle"))
        x_angle_layout.addWidget(self.x_angle_spinbox)
        x_angle_widget.setLayout(x_angle_layout)

        y_angle_widget = QWidget()
        y_angle_layout = QVBoxLayout()
        y_angle_layout.addWidget(QLabel("Y Angle"))
        y_angle_layout.addWidget(self.y_angle_spinbox)
        y_angle_widget.setLayout(y_angle_layout)

        ap_position_widget = QWidget()
        ap_position_layout = QVBoxLayout()
        ap_position_layout.addWidget(QLabel("AP Position"))
        ap_position_layout.addWidget(self.ap_position_spinbox)
        ap_position_widget.setLayout(ap_position_layout)

        self.viewer.window.add_dock_widget(
            [
                x_angle_widget,
                y_angle_widget,
                ap_position_widget,
            ],
            area="bottom",
            name="Slice Options",
        )

        self.viewer.window.add_dock_widget(
            [
                self.progress_bar,
                QLabel("Region"),
                self.region_selection,
                self.mask_button,
                self.next_button,
                self.previous_button,
                self.finish_button,
            ],
            area="left",
            name="Controls",
        )

        self.scan_input()

        self.prior_alignment = False
        self.load_alignment()

        self.predict_sample_slices()
        print("Awaiting fine tuning...", flush=True)
        self.start_viewer()

    def scan_input(self):
        """Scan the input path for valid images and add to file_list"""
        img_ext = [".png", ".jpg", ".jpeg"]
        self.file_list = [
            name
            for name in os.listdir(self.input_path)
            if os.path.isfile(Path(self.input_path) / name)
            and not name.startswith(".")
            and name.endswith(tuple(img_ext))
        ]
        self.file_list.sort()
        self.num_slices = len(self.file_list)
        print(4 + self.num_slices, flush=True)
        print("Scanned input path for images...", flush=True)
        self.progress_bar.setRange(1, self.num_slices)
        self.progress_bar.setValue(1)
        self.progress_bar.setFormat(f"1 / {self.num_slices}")

    def load_alignment(self):
        """Check the input path for a saved alignment pkl"""
        try:
            with open(Path(self.input_path) / "alignment.pkl", "rb") as f:
                self.atlas_slices = pickle.load(f)
            self.prior_alignment = True
            print("Found prior alignment!")
        except:
            print("No comptabile alignment found...")

    def predict_sample_slices(self):
        """Predict the positions of the samples using the tissue predictor"""
        print("Making predictions...", flush=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tissue_predictor = TissuePredictor()
        tissue_predictor.load_state_dict(
            torch.load(self.model_path, map_location=device)
        )
        tissue_predictor.to(device)
        tissue_predictor.eval()

        def restore_label(self, label):
            pos, x_angle, y_angle = label
            # restore target values
            pos_max = 1324
            pos_min = 0
            pos = pos * (pos_max - pos_min) + pos_min
            x_angle_max = 10
            x_angle_min = -10
            x_angle = x_angle * (x_angle_max - x_angle_min) + x_angle_min
            y_angle_max = 10
            y_angle_min = -10
            y_angle = y_angle * (y_angle_max - y_angle_min) + y_angle_min
            return [pos, x_angle, y_angle]

        with torch.no_grad():
            for i in range(self.num_slices):
                # Check if we already loaded a slice with the same name
                if self.file_list[i] in self.atlas_slices.keys():
                    continue

                sample_img = cv2.imread(
                    str(Path(self.input_path) / self.file_list[i]),
                    cv2.IMREAD_GRAYSCALE,
                )
                sample_img = cv2.resize(sample_img, (512, 512))
                sample_img = sample_img.astype(np.float32) / 255.0
                sample_img = torch.from_numpy(sample_img).unsqueeze(0).unsqueeze(0)
                sample_img = sample_img.to(device)
                pred = tissue_predictor(sample_img)
                pred = pred.cpu().numpy()

                # restore pred to regular space
                pred = restore_label(self, pred[0])
                predicted_slice = AtlasSlice(
                    self.file_list[i],
                    pred[0],
                    pred[1],
                    pred[2],
                    self.is_whole,
                )
                predicted_slice.set_slice(self.atlas, self.annotation)
                self.atlas_slices[self.file_list[i]] = predicted_slice

    def save_alignment(self):
        """Save the slices to a pickle file"""
        with open(Path(self.input_path) / "alignment.pkl", "wb") as f:
            pickle.dump(self.atlas_slices, f)

    def update_mask(self):
        """Update the mask of the current slice"""
        self.atlas_slices[self.file_list[self.current_section]].set_mask()

    def update_display(self):
        """Update the viewer to current section"""
        self.viewer.grid.enabled = False
        sample_img = cv2.imread(
            str(Path(self.input_path) / self.file_list[self.current_section]),
            cv2.IMREAD_GRAYSCALE,
        )
        sample_img = cv2.resize(sample_img, (1920, 1080))
        self.tissue_layer.data = sample_img
        # resize atlas to match tissue
        self.atlas_slices[self.file_list[self.current_section]].set_slice(
            self.atlas, self.annotation
        )
        temp_data = cv2.resize(
            self.atlas_slices[self.file_list[self.current_section]].image,
            (1920, 1080),
        )
        self.atlas_layer.data = temp_data
        self.viewer.grid.enabled = True

        # Set the angles and position
        self.x_angle_spinbox.setValue(
            self.atlas_slices[self.file_list[self.current_section]].x_angle
        )
        self.y_angle_spinbox.setValue(
            self.atlas_slices[self.file_list[self.current_section]].y_angle
        )
        self.ap_position_spinbox.setValue(
            self.atlas_slices[self.file_list[self.current_section]].ap_position
        )

    def set_all_angles(self):
        """Update every slice with the current angles"""
        for slice in self.atlas_slices.values():
            slice.x_angle = self.x_angle_spinbox.value()
            slice.y_angle = self.y_angle_spinbox.value()
            slice.set_slice(self.atlas, self.annotation)

    def update_slice(self):
        """Update the angles and region of the current slice"""
        current_slice = self.atlas_slices[self.file_list[self.current_section]]
        current_slice.x_angle = self.x_angle_spinbox.value()
        current_slice.y_angle = self.y_angle_spinbox.value()
        current_slice.region = self.region_tags[self.region_selection.currentText()]
        current_slice.set_slice(self.atlas, self.annotation)
        self.update_display()

    def update_position(self):
        """Update the position of the current slice"""
        current_slice = self.atlas_slices[self.file_list[self.current_section]]
        current_slice.ap_position = self.ap_position_spinbox.value()
        current_slice.set_slice(self.atlas, self.annotation)
        self.update_display()

    def adjust_positions(self):
        """Adjust the positions of all slices based on trend in visted slices"""
        if not self.prior_alignment:
            visted_positions = []
            for i in range(self.visited):
                visted_positions.append(
                    self.atlas_slices[self.file_list[i]].ap_position,
                )

            if len(visted_positions) < 2:
                return

            # fit a line to the visited positions
            x = np.arange(len(visted_positions))
            m, b = np.polyfit(x, visted_positions, 1)

            # adjust the positions of all slices after the visited slices
            for i in range(self.visited, self.num_slices):
                self.atlas_slices[self.file_list[i]].ap_position = m * i + b

    def next_section(self):
        """Move to next section"""
        if self.current_section < self.num_slices - 1:
            self.current_section += 1
            self.visited = max(self.visited, self.current_section)
            self.progress_bar.setValue(self.current_section + 1)
            self.progress_bar.setFormat(
                f"{self.current_section + 1} / {self.num_slices}"
            )
            self.adjust_positions()
            self.update_display()

    def previous_section(self):
        """Move to previous section"""
        if self.current_section > 0:
            self.current_section -= 1
            self.progress_bar.setValue(self.current_section + 1)
            self.progress_bar.setFormat(
                f"{self.current_section + 1} / {self.num_slices}"
            )
            self.adjust_positions()
            self.update_display()

    def finish(self):
        """Finish alignment"""
        # disconnect signals
        self.x_angle_spinbox.valueChanged.disconnect(self.update_slice)
        self.y_angle_spinbox.valueChanged.disconnect(self.update_slice)
        self.ap_position_spinbox.valueChanged.disconnect(self.update_position)
        self.region_selection.currentIndexChanged.disconnect(self.update_slice)
        self.next_button.clicked.disconnect(self.next_section)
        self.previous_button.clicked.disconnect(self.previous_section)
        self.finish_button.clicked.disconnect(self.finish)

        # save alignment
        self.save_alignment()

        # warp images
        print("Warping images...", flush=True)
        # Check for any non-all regions
        regions = np.unique([slice.region for slice in self.atlas_slices.values()])
        if len(regions) > 1:
            other_atlases = {}
            other_annotations = {}

            other_atlases["NC"] = nrrd.read(
                Path(self.nrrd_path) / "ara_nissl_10_no_cerebrum.nrrd", index_order="C"
            )[0]
            other_atlases["C"] = nrrd.read(
                Path(self.nrrd_path) / "ara_nissl_10_cerebrum.nrrd", index_order="C"
            )[0]
            other_annotations["NC"] = nrrd.read(
                Path(self.nrrd_path) / "annotation_10_no_cerebrum.nrrd",
                index_order="C",
            )[0]
            other_annotations["C"] = nrrd.read(
                Path(self.nrrd_path) / "annotation_10_cerebrum.nrrd", index_order="C"
            )[0]

        for i in range(self.num_slices):
            print(f"Warping {self.file_list[i]}...", flush=True)
            current_slice = self.atlas_slices[self.file_list[i]]
            sample = cv2.imread(
                str(Path(self.input_path) / self.file_list[i]),
                cv2.IMREAD_GRAYSCALE,
            )

            if current_slice.region != "A":
                current_slice.set_slice(
                    other_atlases[current_slice.region],
                    other_annotations[current_slice.region],
                )

            warped_labels, warped_atlas, color_label = current_slice.get_registered(
                sample,
            )
            cv2.imwrite(
                str(Path(self.output_path) / f"Atlas_{self.file_list[i]}.png"),
                warped_atlas,
            )
            cv2.imwrite(
                str(Path(self.output_path) / f"Label_{self.file_list[i]}.png"),
                color_label,
            )

            # convert sample to color
            sample = cv2.cvtColor(sample, cv2.COLOR_GRAY2RGB)
            # convert color_label to 8 bit
            color_label = cv2.normalize(
                color_label, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
            )
            # composite image
            composite = cv2.addWeighted(
                sample,
                0.5,
                color_label,
                0.5,
                0,
            )
            cv2.imwrite(
                str(Path(self.output_path) / f"Composite_{self.file_list[i]}.png"),
                composite,
            )

            with open(
                Path(self.output_path) / f"Annotation_{self.file_list[i]}.pkl", "wb"
            ) as f:
                pickle.dump(warped_labels, f)

        self.viewer.close()
        print("Done!", flush=True)

    def start_viewer(self):
        """Start the viewer"""
        # enable grid
        self.viewer.show()
        self.update_display()
        napari.run()


if __name__ == "__main__":
    align_controller = AlignmentController(
        args.nrrd.strip(),
        args.input.strip(),
        args.output.strip(),
        args.structures.strip(),
        args.model.strip(),
        args.spacing if args.spacing else None,
        eval(args.whole),
    )
