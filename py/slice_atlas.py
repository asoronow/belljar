from pathlib import Path
import nrrd
import csv
import numpy as np
import pickle, os
import cv2
from matplotlib import pyplot as plt
from PIL import Image
from scipy import ndimage
import SimpleITK as sitk
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
    '''Using the atlas and annotations create a new atlas that only has cerebrum regions.'''
    # Load the class map
    classMap = {}
    with open(Path(r"C:\Users\imageprocessing\Documents\belljar\csv\class_map.pkl"), "rb") as f:
        classMap = pickle.load(f)
        # add cerebral cortex

    # Load the atlas
    atlas, _ = nrrd.read(Path(r"C:\Users\imageprocessing\.belljar\nrrd\ara_nissl_10.nrrd"))
    annotation, _ = nrrd.read(Path(r"C:\Users\imageprocessing\.belljar\nrrd\annotation_10.nrrd"))

    # Get the unique labels
    z, y, x = annotation.shape
    
    new_atlas = np.zeros((z, y, x//2))
    new_annotation = np.zeros((z, y, x//2)).astype(np.uint32)
    for i in range(z):
        label = annotation[i, :, :x//2]
        section = atlas[i, :, :x//2]

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

    nrrd.write(r"C:\Users\imageprocessing\.belljar\nrrd\ara_nissl_10_all.nrrd", new_atlas, index_order="C", compression_level=1)
    nrrd.write(r"C:\Users\imageprocessing\.belljar\nrrd\annotation_10_all.nrrd", new_annotation, index_order="C", compression_level=1)

def get_normal_vector(tilt_x_degrees, tilt_y_degrees):
    """
    Compute a normal vector with specified tilts around the x and y axes.

    Args:
    - tilt_x_degrees (float): Tilt around the x-axis in degrees.
    - tilt_y_degrees (float): Tilt around the y-axis in degrees.

    Returns:
    - np.array: Normal vector with the specified tilts.
    """
    
    # Convert degrees to radians
    tilt_x = np.radians(tilt_x_degrees)
    tilt_y = np.radians(tilt_y_degrees)
    
    # Initial vector pointing up along the z-axis
    vector = np.array([0, 0, 1])
    
    # Apply tilt around y-axis
    vector = np.array([np.cos(tilt_y) * vector[0] + np.sin(tilt_y) * vector[2],
                       vector[1],
                       -np.sin(tilt_y) * vector[0] + np.cos(tilt_y) * vector[2]])

    # Apply tilt around x-axis
    vector = np.array([vector[0],
                       np.cos(tilt_x) * vector[1] - np.sin(tilt_x) * vector[2],
                       np.sin(tilt_x) * vector[1] + np.cos(tilt_x) * vector[2]])
    
    return vector

def reslice_with_plane(image, normal_vector, point_on_plane):
    """
    Reslice a 3D SimpleITK image using an arbitrary plane defined by a normal_vector and a point_on_plane.

    Args:
    - image (SimpleITK.Image): The 3D SimpleITK image to be resliced.
    - normal_vector (list or numpy.ndarray): The normal vector of the reslicing plane.
    - point_on_plane (list or numpy.ndarray): A point on the reslicing plane.

    Returns:
    - SimpleITK.Image: 2D resliced image.
    """
    # Ensure image is sitk.Image
    if not isinstance(image, sitk.Image):
        image = sitk.GetImageFromArray(image)


    # Make sure the normal vector is a unit vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    
    # Compute the rotation to align the plane with the canonical xy-plane
    current_normal = [0, 0, 1]
    rotation_axis = np.cross(normal_vector, current_normal)
    rotation_angle = np.arccos(np.dot(normal_vector, current_normal))

    # Define the rotation matrix
    rotation_matrix = sitk.VersorTransform(rotation_axis.tolist(), rotation_angle)

    # Translate the image so the point of interest is at the origin
    translation = sitk.TranslationTransform(3)
    translation.SetOffset((-np.array(point_on_plane)).tolist())
    
    # Compose the transformations: first translation then rotation
    composite_transform = sitk.CompositeTransform([translation, rotation_matrix])

    # Define the desired size (we want a slice, so the z-dimension is 1)
    desired_size = list(image.GetSize())
    desired_size[2] = 1  # only one slice

    # Resample the image using the composite transform
    resliced_image = sitk.Resample(image, 
                                   desired_size, 
                                   composite_transform, 
                                   sitk.sitkLinear, 
                                   image.GetOrigin(), 
                                   image.GetSpacing(), 
                                   image.GetDirection(), 
                                   0,  # background value
                                   image.GetPixelID())

    return resliced_image

if __name__ == "__main__":
    atlas, _ = nrrd.read(Path(r"C:\Users\imageprocessing\.belljar\nrrd\ara_nissl_10_all.nrrd"), index_order="C")
    annotation, _ = nrrd.read(Path(r"C:\Users\imageprocessing\.belljar\nrrd\annotation_10_all.nrrd"), index_order="C")


    normal_vector = get_normal_vector(5, 0)
   
    # Define a point on the plane
    point_on_plane = [0, 0, 0]

    # Reslice the image
    resliced_image = reslice_with_plane(atlas, normal_vector, point_on_plane)
    sitk.Show(resliced_image)
    new_atlas = sitk.GetArrayFromImage(resliced_image)

    cv2.imshow("atlas", new_atlas[800, :, :])
    cv2.imshow("old atlas", atlas[800, :, :])
    cv2.waitKey(0)