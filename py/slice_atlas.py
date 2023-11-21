from pathlib import Path
import nrrd
import numpy as np
import pickle
import cv2
from scipy.ndimage import map_coordinates
from skimage import measure
import json
from uuid import uuid4

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
    x, y = np.meshgrid(np.arange(volume.shape[2]), np.arange(volume.shape[1]))

    # Adjust z-position based on tilt angles
    # Ensure data type is float to handle decimal computations
    z = (z_position + x * np.tan(x_angle_rad) + y * np.tan(y_angle_rad)).astype(
        np.float32
    )
    coords = np.array([z, y, x])

    # Extract slice using nearest-neighbor interpolation
    slice_2d = map_coordinates(volume, coords, order=0)

    return slice_2d


def make_angled_data(samples, atlas):
    metadata = {}
    output_path = Path("C:/Users/asoro/Desktop/angled_data/")
    try:
        metadata = pickle.load(open(output_path / "metadata.pkl", "rb"))
    except FileNotFoundError:
        pass

    for _ in range(samples):
        x_angle, y_angle = np.random.rand(2) * 10 - 5  # angles are now between -5 and 5 degrees
        scale = np.random.rand() * 0.4 + 0.8  # scale factor between 0.8 and 1.2 for slight scaling
        shear = np.random.rand() * 0.2 - 0.1  # shear factor between -0.1 and 0.1 for slight skew
        name = str(uuid4())
        
        pos = np.random.randint(100, atlas.shape[0] - 100)
        sample = slice_3d_volume(atlas, pos, x_angle, y_angle)

        if np.random.rand() > 0.5:
            sample = sample[:, : sample.shape[1] // 2]

        # Apply rotation
        center = (sample.shape[1]//2, sample.shape[0]//2)
        rotation_matrix = cv2.getRotationMatrix2D(center, np.random.uniform(-10, 10), scale)
        sample = cv2.warpAffine(sample, rotation_matrix, (sample.shape[1], sample.shape[0]))

        # Apply shear
        M = np.float32([
            [1, shear, 0],
            [0, 1, 0]
        ])
        sample = cv2.warpAffine(sample, M, (sample.shape[1], sample.shape[0]))

        # Resize to target dimensions
        sample = cv2.resize(sample, (256, 256), interpolation=cv2.INTER_LINEAR)
        
        metadata[name] = {
            "x_angle": x_angle,
            "y_angle": y_angle,
            "pos": pos,
            "scale": scale,
            "shear": shear
        }

        # Save to disk
        cv2.imwrite(str(output_path / f"{name}.png"), sample)

    pickle.dump(metadata, open(output_path / "metadata.pkl", "wb"))

def dump_structure_data():
    """Dumps structure graph data to a pickle"""
    project_dir = Path("/Users/alec/Projects/belljar")
    structure_graph_path = project_dir / "csv/structure_graph.json"
    with open(structure_graph_path, "r") as f:
        structure_graph = json.load(f)

    # Get the unique labels
    structure_map = {}

    def hex_to_rgb(hex):
        hex = hex.lstrip("#")
        hlen = len(hex)
        return tuple(int(hex[i : i + hlen // 3], 16) for i in range(0, hlen, hlen // 3))

    def flatten_graph(graph, id_path=[]):
        structure_map[np.uint32(graph["id"])] = {
            "name": graph["name"],
            "acronym": graph["acronym"],
            "color": hex_to_rgb(graph["color_hex_triplet"]),
            "id_path": "/".join(id_path + [str(graph["id"])]),
        }
        current_id_path = id_path.copy() + [str(graph["id"])]
        if "children" in graph:
            for child in graph["children"]:
                flatten_graph(child, current_id_path)

    flatten_graph(structure_graph)

    structure_map[0] = {
        "name": "Lost in Warp",
        "acronym": "LIW",
        "color": (0, 0, 0),
        "id_path": "0",
    }

    with open(project_dir / "csv/structure_map.pkl", "wb") as f:
        pickle.dump(structure_map, f)


def add_outlines(annotation_slice, color_annotation_slice):
    """
    Add a white outline between each unique label in the annotation slice.

    Args:
        annotation_slice (numpy.ndarray): The annotation slice.
        color_annotation_slice (numpy.ndarray): The color annotation slice.

    Returns:
        numpy.ndarray: The color annotation slice with outlines.
    """

    # Get the unique labels
    ids = np.unique(annotation_slice)

    # Get the contours for each label
    contours = []
    for i in ids:
        contours += measure.find_contours(annotation_slice == i, 0.5)

    # Draw the contours
    for i, contour in enumerate(contours):
        color_annotation_slice[np.int32(contour[:, 0]), np.int32(contour[:, 1])] = 0

    return color_annotation_slice


def mask_slice_by_region(atlas_slice, annotation_slice, structure_map, region="C"):
    """
    Mask a slice from the atlas by region for alignment. Currrently only supports cerebrum or non-cerebrum.

    Args:
        atlas_slice (numpy.ndarray): The atlas slice.
        annotation_slice (numpy.ndarray): The annotation slice.
        structure_map (dict): The structure map.
        region (str): The region to mask by. Either 'C' for cerebrum or 'NC' for non-cerebrum.

    Returns:
        numpy.ndarray: The masked atlas slice.
        numpy.ndarray: The masked annotation slice.
    """
    masked_annotation = np.zeros(annotation_slice.shape, dtype=np.uint32)
    masked_atlas = np.zeros(atlas_slice.shape, dtype=np.uint8)

    # TODO: Whole map is included to support arbitary exclusion of regions. This is not efficient for a single region as current.
    cerebrum_regions = []
    non_cerebrum_regions = []
    for key, value in structure_map.items():
        parents = ["567", "1009"]
        if any(parent in value["id_path"].split("/") for parent in parents):
            cerebrum_regions.append(key)
        else:
            non_cerebrum_regions.append(key)
    if region == "C":
        for i in cerebrum_regions:
            masked_annotation[annotation_slice == i] = i
            masked_atlas[annotation_slice == i] = atlas_slice[annotation_slice == i]
    elif region == "NC":
        for i in non_cerebrum_regions:
            masked_annotation[annotation_slice == i] = i
            masked_atlas[annotation_slice == i] = atlas_slice[annotation_slice == i]

    return masked_atlas, masked_annotation


def main():
    atlas_path = Path(r"C:\Users\asoro\.belljar\nrrd\atlas_10.nrrd")
    nissl_atlas = nrrd.read(str(atlas_path))[0]
    make_angled_data(25000, nissl_atlas)


if __name__ == "__main__":
    main()
