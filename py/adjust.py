import numpy as np
import argparse
import os, sys
import pickle
from pathlib import Path
from qtpy.QtWidgets import (
    QApplication,
    QMainWindow,
    QGraphicsView,
    QGraphicsScene,
    QVBoxLayout,
    QPushButton,
    QHBoxLayout,
    QWidget,
)
from qtpy.QtGui import QImage, QPixmap, QPainter, QColor
from qtpy.QtCore import Qt


class ImageAnnotationTool(QMainWindow):
    def __init__(self, img_dir, annotation_dir, structure_map):
        super().__init__()

        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.structure_map = structure_map
        self.current_index = 0

        # Load images and annotations
        self.images = sorted(
            [
                os.path.join(img_dir, f)
                for f in os.listdir(img_dir)
                if f.endswith(".png")
            ]
        )
        self.annotations = sorted(
            [
                os.path.join(annotation_dir, f)
                for f in os.listdir(annotation_dir)
                if f.endswith(".pkl")
            ]
        )

        # GUI Components
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Image Annotation Tool")

        layout = QVBoxLayout()

        # GraphicsView for Image and Annotation
        self.img_view = QGraphicsView(self)
        self.anno_view = QGraphicsView(self)

        layout.addWidget(self.img_view)
        layout.addWidget(self.anno_view)

        # Buttons for navigation
        nav_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous", self)
        self.prev_button.clicked.connect(self.prev_image)
        self.next_button = QPushButton("Next", self)
        self.next_button.clicked.connect(self.next_image)
        nav_layout.addWidget(self.prev_button)
        nav_layout.addWidget(self.next_button)
        layout.addLayout(nav_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.show_image()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image()

    def next_image(self):
        if self.current_index < len(self.images) - 1:
            self.current_index += 1
            self.show_image()

    def show_image(self):
        # Show image
        img = QPixmap(self.images[self.current_index])
        scene = QGraphicsScene(self)
        scene.addPixmap(img)
        self.img_view.setScene(scene)

        # Show annotation
        with open(self.annotations[self.current_index], "rb") as f:
            label_data = pickle.load(f)

        anno = QImage(img.width(), img.height(), QImage.Format_ARGB32)
        anno.fill(Qt.transparent)
        painter = QPainter(anno)
        for x in range(img.width()):
            for y in range(img.height()):
                label_value = label_data[y, x]
                color = self.structure_map.get(label_value, {}).get(
                    "color", (255, 255, 255)
                )
                painter.setPen(QColor(*color))
                painter.drawPoint(x, y)
        painter.end()

        scene = QGraphicsScene(self)
        scene.addPixmap(QPixmap.fromImage(anno))
        self.anno_view.setScene(scene)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Allow adjustment of region alignments"
    )
    parser.add_argument(
        "-a",
        "--annotations",
        help="annotation files path",
        default="",
    )
    parser.add_argument(
        "-i",
        "--images",
        help="images path",
        default="",
    )
    parser.add_argument(
        "-s",
        "--structures",
        help="structures map",
    )
    parser.add_argument(
        "-w", "--whole", help="whole brain mode", action="store_true", default=False
    )
    parser.add_argument(
        "-m",
        "--mode",
        help='mode to run in, either "paint" or "affine"',
        default="affine",
    )
    parser.add_argument(
        "-d", "--draw", help="draw new maps", action="store_true", default=False
    )
    args = parser.parse_args()

    images_path = Path(args.images)
    annotations_path = Path(args.annotations)
    structure_map_path = Path(args.structures)
    structure_map = pickle.load(open(structure_map_path, "rb"))

    app = QApplication([])
    window = ImageAnnotationTool(images_path, annotations_path, structure_map)
    window.show()
    print("Done!", flush=True)
    sys.exit(app.exec_())
