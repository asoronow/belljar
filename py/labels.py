from PIL import Image, ImageEnhance
import numpy as np
import os, argparse
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class Label:
    def __init__(self, labelPath, imageExt=".png"):
        self.labelPath = labelPath
        self.imagePath = labelPath.replace(".txt", imageExt)
        self.imageExt = imageExt
        self.objects = self.__get_objects()

    def __get_objects(self):
        """Get all the labeled objects in the image"""
        with open(self.labelPath, "r") as f:
            lines = f.readlines()
            objects = []
            for line in lines:
                item_class, x, y, w, h = line.split(" ")
                # convert relative floats to ints
                image = Image.open(self.imagePath)
                width, height = image.size
                x = int(float(x) * width)
                y = int(float(y) * height)
                w = int(float(w) * width)
                h = int(float(h) * height)
                objects.append((item_class, x, y, w, h))
            return objects

    def get_cropped_objects(self):
        """Get all the objects in the image within 640px squares"""
        # scan the iamge in 640px squares
        image = Image.open(self.imagePath)
        width, height = image.size

        for i in range(0, width, 640):
            for j in range(0, height, 640):
                square_x_max = i + 640
                square_y_max = j + 640
                square_x_min = i
                square_y_min = j
                cropped_image = image.crop(
                    (square_x_min, square_y_min, square_x_max, square_y_max)
                )
                new_objects = []
                for item_class, x, y, w, h in self.objects:
                    if (
                        x > square_x_min
                        and x < square_x_max
                        and y > square_y_min
                        and y < square_y_max
                    ):
                        new_x = x - square_x_min
                        new_y = y - square_y_min

                        new_object = (item_class, new_x, new_y, w, h)
                        new_objects.append(new_object)

                if len(new_objects) == 0:
                    continue

                cropped_image.save(
                    self.imagePath.replace(
                        self.imageExt, f"._{i}_{j}{self.imageExt}"
                    ).replace("\\r176_labels", "\\r176_labels\\cropped"),
                )

                with open(
                    self.imagePath.replace(self.imageExt, f"._{i}_{j}.txt").replace(
                        "\\r176_labels", "\\r176_labels\\cropped"
                    ),
                    "w",
                ) as f:
                    for item_class, x, y, w, h in new_objects:
                        f.write(f"{item_class} {x/640} {y/640} {w/640} {h/640}\n")

    def display_boxes(self):
        """Show the object boxes on the image"""
        image = Image.open(self.imagePath)
        width, height = image.size
        figure = plt.figure()
        ax = figure.add_subplot(111)
        ax.imshow(image, cmap="gray")

        for item_class, x, y, w, h in self.objects:
            x = x - w / 2
            y = y - h / 2
            rect = Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=1)
            ax.add_patch(rect)

        plt.show()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("directory", help="Directory of label files")

    args = args.parse_args()

    for file in os.listdir(args.directory):
        if file.endswith(".txt"):
            label = Label(os.path.join(args.directory, file), imageExt=".tiff")
            label.get_cropped_objects()
