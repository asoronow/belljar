import os
import numpy as np
import cv2
import pickle
from pathlib import Path
from demons import register_to_atlas
from slice_atlas import slice_3d_volume, add_outlines, mask_slice_by_region
from model import TissuePredictor
import nrrd
import SimpleITK as sitk
import torch
from torchvision import transforms
import napari
import copy
import argparse
from qtpy.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QPushButton,
    QProgressBar,
    QLabel,
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QSlider,
    QWidget,
    QMainWindow,
)
from segment_anything import SamPredictor, sam_model_registry
from qtpy import QtCore, QtGui
from qtpy.QtCore import QTimer
from adjust import numpy_array_to_qimage
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class ImageEraser(QMainWindow):
    closed = QtCore.Signal()

    def __init__(self, image):
        super().__init__()
        self.image = image
        self.mask_image = np.zeros_like(self.image)
        self.drawing = False
        self.brush_size = 3
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Image Eraser")
        container = QWidget()
        ui_layout = QVBoxLayout()
        self.img_view = QGraphicsView(self)
        self.img_view.setMouseTracking(True)
        self.img_view.viewport().installEventFilter(self)

        self.img_scene = QGraphicsScene(self)
        self.qimg = numpy_array_to_qimage(self.image)
        self.img_pixmap = QtGui.QPixmap.fromImage(self.qimg)
        self.img_scene.addPixmap(self.img_pixmap)
        self.img_view.setScene(self.img_scene)
        # Slider for brush size
        self.brush_size_slider = QSlider(QtCore.Qt.Horizontal, self)
        # Set label
        self.brush_size_slider_label = QLabel("Brush Size")
        self.brush_size_slider_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeading)
        self.brush_size_slider.setMinimum(1)
        self.brush_size_slider.setMaximum(10)
        self.brush_size_slider.setValue(self.brush_size)
        self.brush_size_slider.valueChanged.connect(self.update_brush_size)

        # Buttons
        self.save_button = QPushButton("Save", self)
        self.cancel_button = QPushButton("Cancel", self)
        self.save_button.clicked.connect(self.save_mask)
        self.cancel_button.clicked.connect(self.cancel_changes)

        ui_layout.addWidget(self.img_view)
        ui_layout.addWidget(self.brush_size_slider_label)
        ui_layout.addWidget(self.brush_size_slider)
        ui_layout.addWidget(self.save_button)
        ui_layout.addWidget(self.cancel_button)
        container.setLayout(ui_layout)
        self.setCentralWidget(container)

    def eventFilter(self, source, event):
        if source is self.img_view.viewport():
            if event.type() == QtCore.QEvent.MouseMove and self.drawing:
                self.draw_on_image(event.pos())
                return True  # Indicate that the event is handled
            elif (
                event.type() == QtCore.QEvent.MouseButtonPress
                and event.button() == QtCore.Qt.LeftButton
            ):
                self.drawing = True
                self.draw_on_image(event.pos())
                return True
            elif (
                event.type() == QtCore.QEvent.MouseButtonRelease
                and event.button() == QtCore.Qt.LeftButton
            ):
                self.drawing = False
                return True

        # Call base class method to continue normal event processing
        return super().eventFilter(source, event)

    def draw_on_image(self, qpoint):
        # Convert QGraphicsView coordinates to image coordinates
        image_point = self.img_view.mapToScene(qpoint).toPoint()
        if image_point:
            # Calculate the points to draw using a helper function
            points_to_draw = self.points_in_circle(
                (image_point.x(), image_point.y()), self.brush_size * 2
            )

            # Draw on the mask and image
            painter = QtGui.QPainter(self.img_pixmap)
            pen = QtGui.QPen(
                QtGui.QColor(255, 0, 0), self.brush_size * 2, cap=QtCore.Qt.RoundCap
            )  # *2 for diameter
            painter.setPen(pen)
            for pt in points_to_draw:
                try:
                    # Draw red point on the image
                    painter.drawPoint(pt[0], pt[1])
                    # Set corresponding point in the mask
                    self.mask_image[pt[1], pt[0]] = 1
                except IndexError:
                    # Ignore any out of bounds points
                    pass
            painter.end()

            # Update the scene to reflect the changes
            self.img_scene.update()

            self.update_image()

    def points_in_circle(self, center, radius):
        """Return a list of points in a circle"""
        points = []
        for x in range(center[0] - radius, center[0] + radius + 1):
            for y in range(center[1] - radius, center[1] + radius + 1):
                if (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius**2:
                    points.append((x, y))
        return points

    def update_image(self):
        # Update the QGraphicsScene with the new QPixmap
        self.img_scene.clear()
        self.img_scene.addPixmap(self.img_pixmap)
        self.img_view.setScene(self.img_scene)

    def update_brush_size(self, value):
        self.brush_size = value

    def save_mask(self):
        self.mask_image = self.mask_image.astype(np.uint8)
        # close holes
        kernel = np.ones((5, 5), np.uint8)
        # dilate
        self.mask_image = cv2.dilate(self.mask_image, kernel, iterations=5)
        # erode
        self.mask_image = cv2.erode(self.mask_image, kernel, iterations=5)
        # invert
        self.mask_image = np.logical_not(self.mask_image).astype(np.uint8)
        self.close()

    def cancel_changes(self):
        self.mask_image = np.zeros_like(self.image)
        self.close()

    def closeEvent(self, event):
        self.closed.emit()
        event.accept()


class AtlasSlice:
    """
    Helper object to manage atlas slices

    Parameters:
        section_name (str): the filename of the slice
        ap_position (int): the ap position of the slice
        x_angle (float): the x angle of the slice
        y_angle (float): the y angle of the slice
        region (str): the region of the slice
        hemisphere (str): the hemisphere of the slice
    """

    def __init__(
        self, section_name, ap_position, x_angle, y_angle, region="A", hemisphere="W"
    ):
        self.section_name = section_name
        self.ap_position = int(ap_position)
        self.x_angle = float(x_angle)
        self.y_angle = float(y_angle)
        self.linked = True
        self.region = region
        self.hemisphere = hemisphere
        self.image = None
        self.sam_image = None
        self.label = None
        self.mask = None
        self.eraser_window = None

    def set_mask(self):
        """Set the mask of the slice"""
        self.eraser_window = ImageEraser(self.image)
        self.eraser_window.show()

        # on exit
        self.eraser_window.closed.connect(self.on_exit)

    def on_exit(self):
        self.mask = self.eraser_window.mask_image

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
            try:
                self.image = self.image * self.mask
                self.label = self.label * self.mask
            except:
                self.mask = None
                print("Bad mask! Reset next alignment.")

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
        sam_path (str): path to SAM model
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
        sam_path,
        spacing=None,
        is_whole=True,
        use_legacy=False,
    ):
        self.nrrd_path = nrrd_path
        self.input_path = input_path
        self.output_path = output_path
        self.structures_path = structures_path
        self.model_path = model_path
        self.sam_path = Path(sam_path).expanduser()
        self.spacing = spacing
        self.is_whole = is_whole
        self.use_legacy = use_legacy
        self.viewer = napari.Viewer(
            title="Atlas Alignment",
        )

        atlas_name = "reconstructed_atlas.nrrd" if use_legacy else "atlas_10.nrrd"
        annotation_name = (
            "reconstructed_annotation.nrrd" if use_legacy else "annotation_10.nrrd"
        )

        self.atlas = nrrd.read(
            Path(self.nrrd_path) / atlas_name,
        )[0]
        self.annotation = nrrd.read(
            Path(self.nrrd_path) / annotation_name,
        )[0]

        if not self.is_whole:
            self.atlas = self.atlas[:, :, : self.atlas.shape[2] // 2]
            self.annotation = self.annotation[:, :, : self.annotation.shape[2] // 2]

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

        self.file_list = []
        self.num_slices = 0
        self.atlas_slices = {}

        self.visited = 0  # The index of the furthest visited section
        self.current_section = 0  # The index of the current section
        self.initial_pos = None  # The first section actually selected by the user
        self.predicted_delta = None  # The predicted delta between sections

        self.x_angle_spinbox = QDoubleSpinBox()
        self.x_angle_spinbox.setRange(-10, 10)
        self.x_angle_spinbox.setSingleStep(0.1)
        self.x_angle_spinbox.setSuffix("°")
        self.x_angle_spinbox.valueChanged.connect(self.que_update_slice)

        self.y_angle_spinbox = QDoubleSpinBox()
        self.y_angle_spinbox.setRange(-10, 10)
        self.y_angle_spinbox.setSingleStep(0.1)
        self.y_angle_spinbox.setSuffix("°")
        self.y_angle_spinbox.valueChanged.connect(self.que_update_slice)

        self.link_angles_button = QCheckBox("Link Angles")
        self.link_angles_button.setChecked(True)
        self.link_angles_button.stateChanged.connect(self.update_linkage)

        self.ap_position_spinbox = QDoubleSpinBox()
        if not self.use_legacy:
            self.ap_position_spinbox.setRange(0, 1319)
        else:
            self.ap_position_spinbox.setRange(0, 528)
        self.ap_position_spinbox.setSingleStep(1)
        # no decimal places
        self.ap_position_spinbox.setDecimals(0)
        self.ap_position_spinbox.valueChanged.connect(self.que_update_position)

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
        self.region_selection.currentIndexChanged.connect(self.que_update_slice)

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

        # Timers
        self.slice_update_timer = QTimer()
        self.pos_update_timer = QTimer()
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
                self.link_angles_button,
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

        if not self.prior_alignment:
            self.predict_sample_slices()
        else:
            self.region_selection.setCurrentIndex(
                list(self.region_tags.values()).index(
                    self.atlas_slices[self.file_list[self.current_section]].region
                )
            )

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
        if self.num_slices == 0:
            print("No images found!", flush=True)
            exit(1)
        self.progress_bar.setRange(1, self.num_slices)
        self.progress_bar.setValue(1)
        self.progress_bar.setFormat(f"1 / {self.num_slices}")

    def load_alignment(self):
        """Check the input path for a saved alignment pkl"""
        try:
            with open(Path(self.input_path) / "alignment.pkl", "rb") as f:
                self.atlas_slices = pickle.load(f)
            self.prior_alignment = True

            # reload slices and refresh class definition
            for _, atlas_slice in self.atlas_slices.items():
                # re-init with old values
                old_name = atlas_slice.section_name
                old_x = atlas_slice.x_angle
                old_y = atlas_slice.y_angle
                old_pos = atlas_slice.ap_position
                old_region = atlas_slice.region
                old_mask = atlas_slice.mask

                self.atlas_slices[old_name] = AtlasSlice(
                    old_name,
                    old_pos,
                    old_x,
                    old_y,
                    region=old_region,
                )
                self.atlas_slices[old_name].mask = old_mask
                self.atlas_slices[old_name].set_slice(self.atlas, self.annotation)

            print("Found prior alignment!")
            # Check if we have any new slices in our input
            # Compare files names to keys in atlas_slices
            new_files = set(self.file_list) - set(self.atlas_slices.keys())
            if new_files:
                print("New slices found, re-predicting...", flush=True)
                self.predict_sample_slices()

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

        def restore_label(label, legacy=False):
            pos, x_angle, y_angle = label
            # restore target values
            pos_max = 1324 if not legacy else 528
            pos_min = 0
            pos = pos * (pos_max - pos_min) + pos_min
            x_angle_max = 10
            x_angle_min = -10
            x_angle = x_angle * (x_angle_max - x_angle_min) + x_angle_min
            y_angle_max = 10
            y_angle_min = -10
            y_angle = y_angle * (y_angle_max - y_angle_min) + y_angle_min
            return [pos, x_angle, y_angle]

        def sobel(image):
            image = cv2.GaussianBlur(image, (3, 3), sigmaX=0, sigmaY=0)
            gX = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3, delta=25)
            gY = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3, delta=25)

            gX = cv2.convertScaleAbs(gX)
            gY = cv2.convertScaleAbs(gY)

            combined = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
            return combined

        with torch.no_grad():
            x_angles = []
            y_angles = []
            positions = []

            for i in range(self.num_slices):
                # Check if we already loaded a slice with the same name
                if self.file_list[i] in self.atlas_slices.keys():
                    x_angles += [self.atlas_slices[self.file_list[i]].x_angle]
                    y_angles += [self.atlas_slices[self.file_list[i]].y_angle]
                    positions += [self.atlas_slices[self.file_list[i]].ap_position]
                    continue

                sample_img = cv2.imread(
                    str(Path(self.input_path) / self.file_list[i]),
                    cv2.IMREAD_GRAYSCALE,
                )
                # match histogram
                sample_img = cv2.resize(sample_img, (256, 256))
                sample_img = sobel(sample_img)
                sample_img = transforms.ToTensor()(sample_img)
                sample_img = transforms.Normalize(mean=0.1253, std=0.0986)(sample_img)

                sample_img = sample_img.unsqueeze(0)
                sample_img = sample_img.to(device)
                pred = tissue_predictor(sample_img)
                pred = pred.cpu().numpy()

                # restore pred to regular space
                pred = restore_label(pred[0], self.use_legacy)
                x_angles.append(pred[1])
                y_angles.append(pred[2])
                positions.append(pred[0])

            average_x = np.mean(x_angles)
            average_y = np.mean(y_angles)

            if self.num_slices > 1:
                delta_pos = np.mean(np.diff(positions))
            else:
                delta_pos = 0
            self.predicted_delta = delta_pos

            hemi = "W" if self.is_whole else "L"

            for i in range(self.num_slices):
                predicted_slice = AtlasSlice(
                    self.file_list[i],
                    positions[i],
                    average_x,
                    average_y,
                    hemisphere=hemi,
                )

                predicted_slice.set_slice(self.atlas, self.annotation)
                self.atlas_slices[self.file_list[i]] = predicted_slice

    def save_alignment(self):
        """Save the slices to a pickle file"""
        # get rid of image and label
        saved_copy = {}
        for section_name, atlas_slice in self.atlas_slices.items():
            atlas_slice.eraser_window = None
            this_copy = copy.deepcopy(atlas_slice)
            this_copy.image = None
            this_copy.label = None
            saved_copy[section_name] = this_copy

        with open(Path(self.input_path) / "alignment.pkl", "wb") as f:
            pickle.dump(saved_copy, f)

    def update_mask(self):
        """Update the mask of the current slice"""
        self.atlas_slices[self.file_list[self.current_section]].set_mask()

    def _find_aspect_constrained_size(self, img1, img2):
        """
        Find the ideal size to resize both images to, ensuring:
        - At least one dimension of each image is at least 1080p.
        - The individual aspect ratios of both images are maintained.
        - The sizes are compatible with one another.

        Parameters:
        - img1: First image (assumed to be a NumPy array or similar with shape (height, width)).
        - img2: Second image (assumed to be a NumPy array or similar with shape (height, width)).

        Returns:
        - Tuple (width, height): Ideal dimensions to resize both images to.
        """

        def calculate_target_size(img):
            height, width = img.shape[:2]
            aspect_ratio = width / height
            if height > width:
                # Height is the larger dimension
                target_height = max(height, 1080)
                target_width = int(target_height * aspect_ratio)
            else:
                # Width is the larger dimension
                target_width = max(width, 1080)
                target_height = int(target_width / aspect_ratio)
            return target_width, target_height

        target_size_img1 = calculate_target_size(img1)
        target_size_img2 = calculate_target_size(img2)

        # The target size should be the max width and max height obtained from the two images
        target_width = max(target_size_img1[0], target_size_img2[0])
        target_height = max(target_size_img1[1], target_size_img2[1])

        return (target_width, target_height)

    def update_display(self):
        """Update the viewer to current section"""
        self.viewer.grid.enabled = False

        sample_img = cv2.imread(
            str(Path(self.input_path) / self.file_list[self.current_section]),
            cv2.IMREAD_GRAYSCALE,
        )

        new_size = self._find_aspect_constrained_size(
            sample_img,
            self.atlas_slices[self.file_list[self.current_section]].image,
        )
        sample_img = cv2.resize(sample_img, new_size)

        self.tissue_layer.data = sample_img
        # resize atlas to match tissue
        self.atlas_slices[self.file_list[self.current_section]].set_slice(
            self.atlas, self.annotation
        )

        temp_data = cv2.resize(
            self.atlas_slices[self.file_list[self.current_section]].image,
            new_size,
        )
        self.atlas_layer.data = temp_data
        self.viewer.grid.enabled = True

        # Set linkage
        self.link_angles_button.setChecked(
            self.atlas_slices[self.file_list[self.current_section]].linked
        )

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
        self.region_selection.setCurrentIndex(
            list(self.region_tags.values()).index(
                self.atlas_slices[self.file_list[self.current_section]].region
            )
        )

        self.mask_button.setText(
            "Set Mask"
            if self.atlas_slices[self.file_list[self.current_section]].mask is None
            else "Update Mask"
        )

    def update_linkage(self):
        """Update the linkage of the current slice"""
        self.atlas_slices[self.file_list[self.current_section]].linked = (
            self.link_angles_button.isChecked()
        )

    def set_all_angles(self):
        """Update every slice with the current angles"""
        # Check if current slice is linked
        current_slice = self.atlas_slices[self.file_list[self.current_section]]
        if current_slice.linked:
            for this_slice in self.atlas_slices.values():
                if this_slice.linked:
                    this_slice.x_angle = self.x_angle_spinbox.value()
                    this_slice.y_angle = self.y_angle_spinbox.value()
            self.update_display()

    def que_update_slice(self):
        """Queue an update to the display"""
        # Check if timer active
        if self.slice_update_timer.isActive():
            self.slice_update_timer.stop()

        self.slice_update_timer.singleShot(500, self.update_slice)

    def que_update_position(self):
        """Queue an update to the display"""
        # Check if timer active
        if self.pos_update_timer.isActive():
            self.pos_update_timer.stop()

        self.pos_update_timer.singleShot(500, self.update_position)

    def update_slice(self):
        """Update the angles and region of the current slice"""
        current_slice = self.atlas_slices[self.file_list[self.current_section]]
        current_slice.x_angle = self.x_angle_spinbox.value()
        current_slice.y_angle = self.y_angle_spinbox.value()
        current_slice.region = self.region_tags[self.region_selection.currentText()]
        current_slice.set_slice(self.atlas, self.annotation)
        self.set_all_angles()
        self.update_display()

    def update_position(self):
        """Update the position of the current slice"""
        current_slice = self.atlas_slices[self.file_list[self.current_section]]
        current_slice.ap_position = self.ap_position_spinbox.value()
        current_slice.set_slice(self.atlas, self.annotation)
        self.update_display()

    def adjust_positions(self):
        """Adjust the positions of all slices based on trend in visited slices"""
        if not self.prior_alignment and self.visited < self.num_slices - 1:
            visited_positions = []
            for i in range(self.visited):
                visited_positions.append(
                    self.atlas_slices[self.file_list[i]].ap_position,
                )

            if len(visited_positions) < 2:
                return

            degree = 2
            poly_model = make_pipeline(
                StandardScaler(), PolynomialFeatures(degree), Ridge()
            )

            x = np.arange(len(visited_positions)).reshape(-1, 1)
            y = np.array(visited_positions)

            # Fit the model
            poly_model.fit(x, y)

            # Use the model for predictions
            x_predict = np.arange(self.visited, self.num_slices).reshape(-1, 1)
            predictions = poly_model.predict(x_predict)

            # Adjust positions
            for i, new_position in enumerate(predictions, start=self.visited):
                if self.spacing is not None:
                    self.atlas_slices[self.file_list[i]].ap_position = max(
                        new_position, int(self.spacing) // 10
                    )
                else:
                    self.atlas_slices[self.file_list[i]].ap_position = new_position

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


    def isolate_section(self, sample):
        """
        Use SAM to allow the user to isolate each section in the image
        Args:
            sample: image to isolate (gray scale, uint8)
        """

        # Load SAM model
        sam = sam_model_registry["vit_b"](checkpoint=self.sam_path)

        # Check for CUDA or MPS and move the model to the appropriate device
        if torch.cuda.is_available():
            sam = sam.to(device="cuda")
        elif torch.backends.mps.is_available():
            sam = sam.to(device="mps")

        # Set the image for SAM
        predictor = SamPredictor(sam)
        sample_image = cv2.cvtColor(sample.copy(), cv2.COLOR_GRAY2BGR)
        predictor.set_image(sample_image)

        # Prepare for point selection
        points = [[sample.shape[1] // 2, sample.shape[0] // 2]]

        # def get_point(event, x, y, flags, param):
        #     if event == cv2.EVENT_LBUTTONDOWN:
        #         if len(points) < 3:  # Limit to 3 points
        #             points.append([x, y])
        #             cv2.circle(sample_image, (x, y), 3, (0, 0, 255), -1)
        #             cv2.imshow(f"Point Selector", sample_image)

        # # Create a window to display the image and set the mouse callback
        # cv2.namedWindow("Section Isolation")
        # cv2.setMouseCallback("Section Isolation", get_point)
        # cv2.imshow("Section Isolation", sample_image)
        # cv2.waitKey(0)
        # cv2.destroyWindow("Section Isolation")

        if len(points) > 0:
            # Convert points to numpy array
            points_np = np.array(points)

            # Generate mask using SAM
            masks, _, _ = predictor.predict(points_np, np.array([1] * len(points_np)))

            # Display the generated mask for confirmation
            mask = masks[0]  # Assuming the first mask is the most relevant
            return (sample * mask.astype(np.uint8))
            # mask_display = mask.astype(np.uint8) * 255
            # # convert to color
            # mask_display = cv2.applyColorMap(mask_display, cv2.COLORMAP_JET)
            # composite = cv2.addWeighted(sample_image, 0.5, mask_display, 0.5, 0)
            # cv2.imshow("Generated Mask (Press 'y' to confirm 'n' to cancel)", composite)
            # key = cv2.waitKey(0)
            # if key == ord('y'):  # User confirms the mask
            #     cv2.destroyAllWindows()
            #     return sample * mask.astype(np.uint8)
            # elif key == ord('n'):  # User cancels the mask
            #     cv2.destroyAllWindows()
            #     self.isolate_section(sample)

    def finish(self):
        """Finish alignment"""
        # disconnect signals
        self.x_angle_spinbox.valueChanged.disconnect(self.que_update_slice)
        self.y_angle_spinbox.valueChanged.disconnect(self.que_update_slice)
        self.ap_position_spinbox.valueChanged.disconnect(self.que_update_position)
        self.region_selection.currentIndexChanged.disconnect(self.que_update_slice)
        self.next_button.clicked.disconnect(self.next_section)
        self.previous_button.clicked.disconnect(self.previous_section)
        self.finish_button.clicked.disconnect(self.finish)

        # save alignment
        self.save_alignment()

        # warp images
        print("Warping images...", flush=True)
        with open(self.structures_path, "rb") as f:
            structure_map = pickle.load(f)

        for i in range(self.num_slices):
            print(f"Warping {self.file_list[i]}...", flush=True)
            current_slice = self.atlas_slices[self.file_list[i]]
            sample = cv2.imread(
                str(Path(self.input_path) / self.file_list[i]),
                cv2.IMREAD_GRAYSCALE,
            )

            # isolate
            # sample = self.isolate_section(sample)

            if current_slice.region != "A":
                masked_atlas, masked_annotation = mask_slice_by_region(
                    current_slice.image,
                    current_slice.label,
                    structure_map,
                    current_slice.region,
                )
                current_slice.image = masked_atlas
                current_slice.label = masked_annotation



            warped_labels, warped_atlas, color_label = current_slice.get_registered(
                sample,
            )

            stripped_filename = self.file_list[i].split(".")
            stripped_filename = ".".join(stripped_filename[:-1])

            cv2.imwrite(
                str(Path(self.output_path) / f"Atlas_{stripped_filename}.png"),
                warped_atlas,
            )
            color_label = add_outlines(warped_labels, color_label)
            # make label rgb
            color_label = cv2.cvtColor(color_label, cv2.COLOR_BGR2RGB)
            cv2.imwrite(
                str(Path(self.output_path) / f"Label_{stripped_filename}.png"),
                color_label,
            )

            # convert sample to color
            sample = cv2.cvtColor(sample, cv2.COLOR_GRAY2RGB)
            # composite image
            composite = cv2.addWeighted(
                sample,
                0.80,
                color_label,
                0.20,
                0,
            )

            cv2.imwrite(
                str(Path(self.output_path) / f"Composite_{stripped_filename}.png"),
                composite,
            )

            with open(
                Path(self.output_path) / f"Annotation_{stripped_filename}.pkl", "wb"
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
    parser = argparse.ArgumentParser(description="Map sections to atlas space")
    parser.add_argument(
        "-o",
        "--output",
        help="output directory, only use if graphical false",
        default="",
    )
    parser.add_argument(
        "-i", "--input", help="input directory, only use if graphical false", default=""
    )
    parser.add_argument("-m", "--model", default="../models/predictor_encoder.pt")
    parser.add_argument("-s", "--sam", default="~/.belljar/models/sam_vit_b.pth")
    parser.add_argument("-n", "--nrrd", help="path to nrrd files", default="")
    parser.add_argument("-w", "--whole", default=False)
    parser.add_argument(
        "-a", "--spacing", help="override predicted spacing", default=False
    )
    parser.add_argument("-l", "--legacy", help="use legacy atlas", default=False)
    parser.add_argument("-c", "--map", help="map file", default="../csv/class_map.pkl")
    args = parser.parse_args()

    align_controller = AlignmentController(
        nrrd_path=args.nrrd.strip(),
        input_path=args.input.strip(),
        output_path=args.output.strip(),
        structures_path=args.map.strip(),
        model_path=args.model.strip(),
        sam_path=args.sam.strip(),
        spacing=args.spacing if args.spacing else None,
        is_whole=eval(args.whole),
        use_legacy=eval(args.legacy),
    )
