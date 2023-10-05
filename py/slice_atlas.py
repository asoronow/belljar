from pathlib import Path
import nrrd
import csv
import numpy as np
import pickle, os
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import map_coordinates
import time

# Path to nrrd
nrrdPath = "C:\\Users\\Alec\\Projects\\aba-nrrd\\raw"


def makeAtlasMap():
    """Creates a map of all the unique labels in the atlas"""
    classMap = {}
    r = 0
    data, _ = nrrd.read(nrrdPath + f"/r_annotation_{r}.nrrd")
    z, x, y = data.shape

    usedColors = []

    def getColor():
        color = np.random.randint(0, 255, (3)).tolist()
        while color in usedColors:
            color = np.random.randint(0, 255, (3)).tolist()

        usedColors.append(color)
        return color

    # open the class csv to build the map
    with open("C:\\Users\\Alec\\Projects\\aba-nrrd\\raw\\class.csv") as csv_file:
        reader = csv.reader(csv_file, delimiter=",")
        header = next(reader)
        itemid = header.index("id")
        id_path = header.index("structure_id_path")
        acronym_idx = header.index("acronym")
        itemname = header.index("name")
        for slice in range(z):
            print(f"Processing slice {slice}", flush=True)
            # get uniques
            uniques = np.unique(data[slice, :, :])
            for label in uniques:
                if label not in classMap:
                    classMap[label] = {}
                    classMap[label]["name"] = ""
                    classMap[label]["acronym"] = ""
                    classMap[label]["id"] = ""
                    classMap[label]["id_path"] = ""

                    if label != 0:
                        for line in reader:
                            if line[itemid] == str(label):
                                classMap[label]["name"] = line[itemname]
                                classMap[label]["acronym"] = line[acronym_idx]
                                classMap[label]["id"] = line[itemid]
                                classMap[label]["id_path"] = line[id_path]
                                classMap[label]["color"] = getColor()
                                break
                        csv_file.seek(0)
                    elif label == 0:
                        classMap[label]["name"] = "background"
                        classMap[label]["acronym"] = "background"
                        classMap[label]["id"] = "0"
                        classMap[label]["id_path"] = "0"
                        classMap[label]["color"] = [0, 0, 0]

        for i, (k, v) in enumerate(classMap.items()):
            classMap[k]["index"] = i
        with open("C:\\Users\\Alec\\Projects\\aba-nrrd\\raw\\classMap.pkl", "wb") as f:
            pickle.dump(classMap, f, pickle.HIGHEST_PROTOCOL)


def createTrainingSet():
    """Make the set of all pngs to train the autoencoder"""
    for r in range(-10, 11, 1):
        print(f"Processing angle {r}", flush=True)
        data, _ = nrrd.read(nrrdPath + f"/r_annotation_{r}.nrrd")
        dataN, _ = nrrd.read(nrrdPath + f"/r_nissl_{r}.nrrd")
        z, x, y = data.shape
        for slice in range(100, z - 100, 1):
            print("Processing slice " + str(slice), flush=True)
            writePathW = nrrdPath + "\\map\\whole" + f"\\r_map_{r}_{slice}.tif"
            writePathH = nrrdPath + "\\map\\half" + f"\\r_map_{r}_{slice}.tif"

            writePathWN = nrrdPath + "\\image\\whole" + f"\\r_nissl_{r}_{slice}.png"
            writePathHN = nrrdPath + "\\image\\half" + f"\\r_nissl_{r}_{slice}.png"

            imageH = np.array(data[slice, :, : y // 2], dtype=np.uint32)
            imageW = np.array(data[slice, :, :], dtype=np.uint32)

            imageHN = np.array(dataN[slice, :, : y // 2], dtype=np.uint16)
            imageWN = np.array(dataN[slice, :, :], dtype=np.uint16)

            # Convert HN and WN to 8 bit
            imageHN = imageHN.astype(np.float32)
            imageWN = imageWN.astype(np.float32)
            imageHN = imageHN / np.max(imageHN)
            imageWN = imageWN / np.max(imageWN)
            imageHN = imageHN * 255
            imageWN = imageWN * 255
            imageHN = imageHN.astype(np.uint8)
            imageWN = imageWN.astype(np.uint8)

            imageHN = Image.fromarray(imageHN)
            imageWN = Image.fromarray(imageWN)
            imageH = Image.fromarray(imageH)
            imageW = Image.fromarray(imageW)

            imageHN = imageHN.resize((256, 256), Image.NEAREST)
            imageWN = imageWN.resize((256, 256), Image.NEAREST)
            imageH = imageH.resize((256, 256), Image.NEAREST)
            imageW = imageW.resize((256, 256), Image.NEAREST)

            imageHN.save(writePathHN)
            imageWN.save(writePathWN)
            imageH.save(writePathH)
            imageW.save(writePathW)


def make_cerebrum_atlas():
    """Using the atlas and annotations create a new atlas that only has cerebrum regions."""
    # Load the class map
    classMap = {}
    with open(
        Path(r"C:\Users\imageprocessing\Documents\belljar\csv\class_map.pkl"), "rb"
    ) as f:
        classMap = pickle.load(f)
        # add cerebral cortex

    # Load the atlas
    atlas, _ = nrrd.read(
        Path(r"C:\Users\imageprocessing\.belljar\nrrd\ara_nissl_10.nrrd")
    )
    annotation, _ = nrrd.read(
        Path(r"C:\Users\imageprocessing\.belljar\nrrd\annotation_10.nrrd")
    )

    # Get the unique labels
    z, y, x = annotation.shape

    new_atlas = np.zeros((z, y, x // 2))
    new_annotation = np.zeros((z, y, x // 2)).astype(np.uint32)
    for i in range(z):
        label = annotation[i, :, : x // 2]
        section = atlas[i, :, : x // 2]

        # Convert section to 8 bit from float 32
        section = cv2.normalize(section, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        out_section = np.zeros((section.shape[0], section.shape[1])).astype(np.uint8)
        out_label = np.zeros((label.shape[0], label.shape[1])).astype(np.uint32)

        out_label = label
        out_section = section

        # for k in range(label.shape[0]):
        #     for j in range(label.shape[1]):
        #         id_path = classMap[label[k, j]]["id_path"].split("/")
        #         # parents = ["567", "776", "896", "768", "484682512"]
        #         parents = ["343", "512", "73", "967", "960", "784","1000", "824"]
        #         for parent in parents:
        #             if parent in id_path:
        #                 out_section[k, j] = section[k, j]
        #                 out_label[k, j] = label[k, j]
        #                 break
        # cv2.imshow("section", out_section)
        # cv2.waitKey(1)
        new_atlas[i, :, :] = out_section
        new_annotation[i, :, :] = out_label

    nrrd.write(
        r"C:\Users\imageprocessing\.belljar\nrrd\ara_nissl_10_all.nrrd",
        new_atlas,
        index_order="C",
        compression_level=1,
    )
    nrrd.write(
        r"C:\Users\imageprocessing\.belljar\nrrd\annotation_10_all.nrrd",
        new_annotation,
        index_order="C",
        compression_level=1,
    )


def slice_3d_volume(volume, z_position, x_angle, y_angle):
    """
    Obtain a slice at a certain point in a 3D volume at an arbitrary angle.

    Args:
        volume (numpy.ndarray): 3D numpy array.
        z_position (int): Position along the z-axis for the slice.
        x_angle (float): Angle in degrees to tilt in the x axis.
        y_angle (float): Angle in degrees to tilt in the y axis.

    Returns:
        numpy.ndarray: 2D sliced array.
    """

    # Convert angles to radians
    x_angle_rad = np.deg2rad(x_angle)
    y_angle_rad = np.deg2rad(y_angle)

    # Create a coordinate grid
    x, y = np.mgrid[0 : volume.shape[1], 0 : volume.shape[2]]

    # Adjust z-position based on tilt angles
    z = z_position + x * np.tan(x_angle_rad) + y * np.tan(y_angle_rad)
    coords = np.array([z, x, y])

    # Extract slice using trilinear interpolation
    slice_2d = map_coordinates(volume, coords, order=1, mode="nearest")

    return slice_2d


if __name__ == "__main__":
    atlas, _ = nrrd.read(
        Path(r"C:\Users\alec\.belljar\nrrd\ara_nissl_10_all.nrrd"), index_order="C"
    )
    annotation, _ = nrrd.read(
        Path(r"C:\Users\alec\.belljar\nrrd\annotation_10_all.nrrd"), index_order="C"
    )
    print("Loaded atlas...")
    # convert atlas to 8 bit
    atlas = cv2.normalize(atlas, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    cv2.namedWindow("image")
    cv2.namedWindow("controls")
    curr_x_angle = 0
    curr_y_angle = 0
    curr_z_position = 0

    def trackbar_callback(x):
        global curr_x_angle, curr_y_angle, curr_z_position
        curr_x_angle = cv2.getTrackbarPos("x_angle", "controls")
        curr_y_angle = cv2.getTrackbarPos("y_angle", "controls")
        curr_z_position = cv2.getTrackbarPos("z_position", "controls")

    cv2.createTrackbar("x_angle", "controls", 0, 10, trackbar_callback)
    cv2.createTrackbar("y_angle", "controls", 0, 10, trackbar_callback)
    cv2.createTrackbar("z_position", "controls", 0, atlas.shape[0], trackbar_callback)
    cv2.setTrackbarMin("z_position", "controls", 0)
    cv2.setTrackbarMin("x_angle", "controls", -10)
    cv2.setTrackbarMin("y_angle", "controls", -10)
    while True:
        left_slice = slice_3d_volume(atlas, curr_z_position, curr_x_angle, curr_y_angle)
        right_slice = slice_3d_volume(
            atlas[:, :, ::-1], curr_z_position, -1 * curr_x_angle, curr_y_angle
        )
        full = np.concatenate((left_slice, right_slice), axis=1)
        cv2.imshow("image", full)
        cv2.waitKey(5)
