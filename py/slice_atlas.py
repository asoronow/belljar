from pathlib import Path
import nrrd
import numpy as np
import pickle
import cv2
from scipy.ndimage import map_coordinates
from skimage import measure
import json


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
    for _ in range(samples):
        x_angle, y_angle = np.random.rand(2)

        # convert to single digit float in range -10 to 10
        x_angle = round((x_angle - 0.5) * 10, 1)
        y_angle = round((y_angle - 0.5) * 10, 1)

        pos = np.random.randint(100, atlas.shape[0] - 100)

        # coin toss sample is whole or hemi
        if np.random.rand() > 0.5:
            sample = slice_3d_volume(atlas, pos, x_angle, y_angle)

            # randomly mirror
            if np.random.rand() > 0.5:
                sample = sample[:, ::-1]

        else:
            sample_l = slice_3d_volume(atlas, pos, x_angle, y_angle)
            sample_r = slice_3d_volume(atlas[:, :, ::-1], pos, -1 * x_angle, y_angle)
            sample = np.concatenate((sample_l, sample_r), axis=1)
        # resize to 640x640
        sample = cv2.resize(sample, (640, 640), interpolation=cv2.INTER_AREA)

        # apply a random rotation
        angle = np.random.randint(-7, 7)
        rot_mat = cv2.getRotationMatrix2D((320, 320), angle, 1.0)
        sample = cv2.warpAffine(sample, rot_mat, (512, 512))
        print(f"Saving sample {pos}_{x_angle}_{y_angle}.png")
        # save to disk
        cv2.imwrite(
            f"C:/Users/asoro/Desktop/angled_data/{pos}_{x_angle}_{y_angle}.png", sample
        )


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
    dump_structure_data()
    # Color
    structure_map = pickle.load(
        open(r"C:\Users\Alec\Projects\belljar\csv\structure_map.pkl", "rb")
    )
    annotation_path = Path(r"C:\Users\Alec\.belljar\nrrd") / "annotation_10.nrrd"
    annotation, _ = nrrd.read(annotation_path)
    atlas_path = Path(r"C:\Users\Alec\.belljar\nrrd") / "atlas_10.nrrd"
    atlas, _ = nrrd.read(atlas_path)
    color_annotation_slice = np.zeros(
        (annotation.shape[1], annotation.shape[2], 3), dtype=np.uint8
    )

    demo_slice = slice_3d_volume(annotation, 700, 5, 0)
    atlas_slice = slice_3d_volume(atlas, 700, 5, 0)
    atlas_slice, demo_slice = mask_slice_by_region(
        atlas_slice, demo_slice, structure_map, region="C"
    )
    ids = np.unique(demo_slice)
    for i in ids:
        try:
            color_annotation_slice[demo_slice == i] = structure_map[i]["color"]
        except:
            pass

    color_annotation_slice = add_outlines(demo_slice, color_annotation_slice)
    cv2.imshow("color", color_annotation_slice)
    cv2.imshow("atlas", atlas_slice)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
