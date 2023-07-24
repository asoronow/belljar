import nrrd
import csv
import numpy as np
import pickle, os
import tifffile
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import interpolation
import nibabel as nib

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


def buildRotatedAtlases(nisslPath, annotationPath, outputPath):
    """Constructions the rotated (z-x) atlases for the most common cutting angles"""
    nData, nHead = nrrd.read(nisslPath)
    aData, aHead = nrrd.read(annotationPath)

    for r in range(-10, 11, 1):
        print(f"Rotating atlas to angle {0}", flush=True)
        nissl_rotatedX = interpolation.rotate(
            nData[:, :, :], angle=r, axes=(0, 2), order=0
        )
        annotation_rotatedX = interpolation.rotate(
            aData[:, :, :], angle=r, axes=(0, 2), order=0
        )
        nrrd.write(str(outputPath) + f"/r_nissl_{r}.nrrd", nissl_rotatedX, nHead)
        nrrd.write(
            str(outputPath) + f"/r_annotation_{r}.nrrd", annotation_rotatedX, aHead
        )


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


def createDAPITrainingSet(atlasPath, labelPath):
    dapiAtlas = nib.load(atlasPath)
    dapiAtlas = dapiAtlas.get_fdata()
    dapiLabels = nib.load(labelPath)
    dapiLabels = dapiLabels.get_fdata()
    # print header
    x, z, y = dapiLabels.shape
    for section in range(z):
        image = Image.fromarray((dapiAtlas[:, section, :] * 30) + 13.21)
        image = image.convert("L")
        image = image.resize((256, 256), Image.Resampling.NEAREST)
        image = image.rotate(90)
        image.save(
            f"C:\\Users\\Alec\\Projects\\dapi-atlas\\image\\whole\\{section}.png"
        )

        labels = dapiLabels[:, section, :]
        labels = labels.astype(np.int32)
        image = Image.fromarray(labels)
        image = image.resize((256, 256), Image.Resampling.NEAREST)
        image = image.rotate(90)
        image.save(f"C:\\Users\\Alec\\Projects\\dapi-atlas\\map\\whole\\{section}.tif")


def createAdjacencyMatrix():
    r = 0
    data, _ = nrrd.read(nrrdPath + f"/r_annotation_{r}.nrrd")
    z, x, y = data.shape
    matrix = np.zeros((1328, 1328))
    with open("mappickle.pkl", "rb") as p:
        classMap = pickle.load(p)
        classMap["997"] = {"index": 1326, "name": "undefined", "color": [0, 0, 0]}
        classMap["0"] = {"index": 1327, "name": "Lost in Warp", "color": [0, 0, 0]}

    for slice in range(0, z, 1):
        section = data[slice, :, :]
        print("\nProcessing slice " + str(slice), flush=True)
        # for each pixel check its adjacent pixels
        for i in range(1, x - 1):
            for j in range(1, y - 1):
                # if the pixel is not background
                if section[i, j] != 0:
                    # check the adjacent pixels
                    for k in range(-1, 2):
                        for l in range(-1, 2):
                            # add an edge between the two
                            matrix[
                                int(classMap[str(section[i, j])]["index"]),
                                int(classMap[str(section[i + k, j + l])]["index"]),
                            ] = 1

    with open("adjacency.pkl", "wb") as p:
        pickle.dump(matrix, p)


if __name__ == "__main__":
    with open("C:\\Users\\Alec\\Projects\\aba-nrrd\\raw\\classMap.pkl", "rb") as p:
        classMap = pickle.load(p)
        for file in os.listdir("C:\\Users\\Alec\\Downloads\\M511-Alignment"):
            if file.endswith(".pkl"):
                # reisze to 256x256
                annotation = pickle.load(
                    open("C:\\Users\\Alec\\Downloads\\M511-Alignment\\" + file, "rb")
                )
                annotation = annotation.astype(np.uint32)
                annotation = Image.fromarray(annotation)
                annotation = annotation.resize((256, 256), Image.NEAREST)
                annotation = np.array(annotation)
                tifffile.imsave(
                    "C:\\Users\\Alec\\Downloads\\M511-Alignment\\" + file[:-4] + ".tif",
                    annotation,
                )
                blank = np.zeros((annotation.shape[0], annotation.shape[1], 3))
                # add color to blank
                print("Processing " + file)
                for i in range(0, blank.shape[0]):
                    for j in range(0, blank.shape[1]):
                        if annotation[i, j] != 0:
                            try:
                                blank[i, j] = classMap[np.int32(annotation[i, j])][
                                    "color"
                                ]
                            except:
                                print("Missing color for " + str(annotation[i, j]))

                blank = blank.astype(np.uint8)
                blank = Image.fromarray(blank)
                blank = blank.resize((256, 256), Image.NEAREST)
                blank.save(
                    "C:\\Users\\Alec\\Downloads\\M511-Alignment\\color\\"
                    + file[:-4]
                    + ".png"
                )

    # usedColors = []

    # def getColor():
    #     color = np.random.randint(0, 255, (3)).tolist()
    #     while color in usedColors:
    #         color = np.random.randint(0, 255, (3)).tolist()

    #     usedColors.append(color)
    #     return color

    # with open("../csv/structure_tree_safe_2017.csv", "r") as f:
    #     reader = csv.reader(f, delimiter=",")
    #     header = next(reader)
    #     itemid = header.index("id")
    #     id_path = header.index("structure_id_path")
    #     acronym_idx = header.index("acronym")
    #     itemname = header.index("name")

    #     classMap = {}
    #     classMap["997"] = {
    #         "name": "Root",
    #         "acronym": "root",
    #         "parent_id": "997",
    #         "color": [0, 0, 0],
    #     }
    #     classMap["0"] = {
    #         "name": "Background",
    #         "acronym": "bkg",
    #         "parent_id": "0",
    #         "color": [0, 0, 0],
    #     }
    #     for index, row in enumerate(reader):
    #         unique_id = row[itemid]
    #         name = row[itemname]
    #         path = row[id_path]
    #         acronym = row[acronym_idx]
    #         parent_ids = path.split("/")[1:-1]  # Extract the parent IDs from the path
    #         if len(parent_ids) > 1:
    #             parent = parent_ids[-2]
    #         else:
    #             parent = "997"
    #         if unique_id not in classMap.keys():
    #             classMap[unique_id] = {
    #                 "name": name,
    #                 "color": getColor(),
    #                 "acronym": acronym,
    #                 "parent_id": parent,
    #             }
