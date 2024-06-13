from pathlib import Path
import nrrd
import numpy as np
import pickle
import cv2
from scipy.ndimage import map_coordinates, affine_transform
from skimage import measure
import json
from uuid import uuid4
from demons import resize_image_nearest_neighbor
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


def make_angled_data(samples, atlas):
    metadata = {}
    output_path = Path("C:/Users/asoro/Desktop/angled_data/")
    try:
        metadata = pickle.load(open(output_path / "metadata.pkl", "rb"))
    except FileNotFoundError:
        pass

    for _ in range(samples):
        x_angle, y_angle = (
            np.random.rand(2) * 10 - 5
        )  # angles are now between -5 and 5 degrees
        scale = (
            np.random.rand() * 0.4 + 0.8
        )  # scale factor between 0.8 and 1.2 for slight scaling
        shear = (
            np.random.rand() * 0.2 - 0.1
        )  # shear factor between -0.1 and 0.1 for slight skew
        name = str(uuid4())

        pos = np.random.randint(100, atlas.shape[0] - 100)
        sample = slice_3d_volume(atlas, pos, x_angle, y_angle)

        if np.random.rand() > 0.5:
            sample = sample[:, : sample.shape[1] // 2]

        # Apply rotation
        center = (sample.shape[1] // 2, sample.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(
            center, np.random.uniform(-10, 10), scale
        )
        sample = cv2.warpAffine(
            sample, rotation_matrix, (sample.shape[1], sample.shape[0])
        )

        # Apply shear
        M = np.float32([[1, shear, 0], [0, 1, 0]])
        sample = cv2.warpAffine(sample, M, (sample.shape[1], sample.shape[0]))

        # Resize to target dimensions
        sample = cv2.resize(sample, (256, 256), interpolation=cv2.INTER_LINEAR)

        metadata[name] = {
            "x_angle": x_angle,
            "y_angle": y_angle,
            "pos": pos,
            "scale": scale,
            "shear": shear,
        }

        # Save to disk
        cv2.imwrite(str(output_path / f"{name}.png"), sample)

    pickle.dump(metadata, open(output_path / "metadata.pkl", "wb"))



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


def is_transform_in_bounds(image, transform_matrix):
    height, width = image.shape[:2]
    corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]])
    new_corners = np.dot(transform_matrix, corners.T).T

    min_x = new_corners[:, 0].min()
    max_x = new_corners[:, 0].max()
    min_y = new_corners[:, 1].min()
    max_y = new_corners[:, 1].max()

    return min_x >= 0 and max_x <= width and min_y >= 0 and max_y <= height


def adjust_brightness_contrast(image, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        image = cv2.addWeighted(image, alpha_c, image, 0, gamma_c)

    return image


def generate_sample(
    i, num_samples, atlas, annotation, experiment_path, metadata_file, ttrs, lock
):
    start_time = time.time()
    x_angle, y_angle = np.random.uniform(-15, 15, 2)
    z_position = np.random.randint(200, 1200)
    sample = slice_3d_volume(atlas, z_position, x_angle, y_angle)
    sample_annotation = slice_3d_volume(annotation, z_position, x_angle, y_angle)

    # 50% chance of only using half of the brain
    if np.random.rand() > 0.5:
        removed_pixels = sample.shape[1] // 2
        sample = sample[:, : sample.shape[1] // 2]
        # recenter by padding the removed pixels
        sample = np.pad(sample, ((0, 0), (0, removed_pixels // 2)), "constant")

    center = (sample.shape[1] // 2, sample.shape[0] // 2)
    rotation_angle = np.random.uniform(-10, 10)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)
    sample = cv2.warpAffine(sample, rotation_matrix, (sample.shape[1], sample.shape[0]))
    # sample_annotation = affine_transform(
    #     sample_annotation, rotation_matrix[:, :2], rotation_matrix[:, 2]
    # )

    shear_y = np.random.rand() * 0.25 - 0.125
    shear_matrix = np.float32([[1, 0, 0], [shear_y, 1, 0]])

    sample = cv2.warpAffine(sample, shear_matrix, (sample.shape[1], sample.shape[0]))
    # sample_annotation = affine_transform(
    #     sample_annotation, shear_matrix[:, :2], shear_matrix[:, 2]
    # ).astype(np.uint32)

    sample = np.pad(sample, 25, "constant", constant_values=0)
    # sample_annotation = np.pad(
    #     sample_annotation, 25, "constant", constant_values=0
    # ).astype(np.uint32)

    sample = cv2.resize(sample, (224, 224), interpolation=cv2.INTER_LINEAR)
    # sample_annotation = resize_image_nearest_neighbor(sample_annotation, (224, 224))

    random_name = str(uuid4())
    sample_filename = f"S_{random_name}.png"
    # annotation_filename = f"S_{random_name}.pkl"

    cv2.imwrite(str(experiment_path / sample_filename), sample)
    # with open(experiment_path / annotation_filename, "wb") as f:
    #     pickle.dump(sample_annotation, f)

    metadata_entry = f"{sample_filename},{x_angle},{y_angle},{z_position}\n"
    with lock:
        with open(metadata_file, "a") as f:
            f.write(metadata_entry)
        
        if len(ttrs) > 50:
            ttrs.pop(0)

        ttrs.append(time.time() - start_time)

        time_remaining = np.mean(ttrs) * (num_samples - i)
        formatted_time_remaining = time.strftime(
            "%H:%M:%S", time.gmtime(time_remaining)
        )
        if i % 3 == 0:
            print(
                f"Samples completed: {i}/{num_samples} | Time remaining: {formatted_time_remaining}",
                end="\r",
            )


def generate_sample_pair(
    i, num_samples, atlas, experiment_path, metadata_file, ttrs, lock
):
    start_time = time.time()
    x_angle, y_angle = np.random.uniform(-15, 15, 2)
    z_position = np.random.randint(200, 1200)
    
    # Generate the original sample
    original_sample = slice_3d_volume(atlas, z_position, x_angle, y_angle)
    
    transformed_sample = original_sample.copy()
    
    is_hemi = False
    # 50% chance of only using half of the brain
    if np.random.rand() > 0.5:
        is_hemi = True
        removed_pixels = transformed_sample.shape[1] // 2
        transformed_sample = transformed_sample[:, : transformed_sample.shape[1] // 2]
        original_sample = original_sample[:, : original_sample.shape[1] // 2]
        # Recenter by padding the removed pixels
        transformed_sample = np.pad(transformed_sample, ((0, 0), (0, removed_pixels // 2)), "constant")
        original_sample = np.pad(original_sample, ((0, 0), (0, removed_pixels // 2)), "constant")
    
    center = (transformed_sample.shape[1] // 2, transformed_sample.shape[0] // 2)
    rotation_angle = np.random.uniform(-10, 10)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1)
    transformed_sample = cv2.warpAffine(transformed_sample, rotation_matrix, (transformed_sample.shape[1], transformed_sample.shape[0]))

    shear_y = np.random.rand() * 0.25 - 0.125
    shear_matrix = np.float32([[1, 0, 0], [shear_y, 1, 0]])
    transformed_sample = cv2.warpAffine(transformed_sample, shear_matrix, (transformed_sample.shape[1], transformed_sample.shape[0]))

    transformed_sample = np.pad(transformed_sample, 25, "constant", constant_values=0)
    transformed_sample = cv2.resize(transformed_sample, (224, 224), interpolation=cv2.INTER_LINEAR)
    original_sample = np.pad(original_sample, 25, "constant", constant_values=0)
    original_sample = cv2.resize(original_sample, (224, 224), interpolation=cv2.INTER_LINEAR)

    # Generate filenames for original and transformed images
    random_name = str(uuid4())
    original_sample_filename = f"S_{random_name}.png"
    transformed_sample_filename = f"S_{random_name}.png"

    # Create subdirectories for original and transformed images
    original_subdir = experiment_path / "original"
    transformed_subdir = experiment_path / "transformed"
    original_subdir.mkdir(parents=True, exist_ok=True)
    transformed_subdir.mkdir(parents=True, exist_ok=True)
    hemi_subdir_original = original_subdir / "hemi"
    hemi_subdir_transformed = transformed_subdir / "hemi"
    whole_subdir_original = original_subdir / "whole"
    whole_subdir_transformed = transformed_subdir / "whole"
    hemi_subdir_original.mkdir(parents=True, exist_ok=True)
    hemi_subdir_transformed.mkdir(parents=True, exist_ok=True)
    whole_subdir_original.mkdir(parents=True, exist_ok=True)
    whole_subdir_transformed.mkdir(parents=True, exist_ok=True)

    # Save images to respective subdirectories
    if is_hemi:
        cv2.imwrite(str(hemi_subdir_original / original_sample_filename), original_sample)
        cv2.imwrite(str(hemi_subdir_transformed / transformed_sample_filename), transformed_sample)
    else:
        cv2.imwrite(str(whole_subdir_original / original_sample_filename), original_sample)
        cv2.imwrite(str(whole_subdir_transformed / transformed_sample_filename), transformed_sample)

    metadata_entry = f"{original_sample_filename},{transformed_sample_filename},{x_angle},{y_angle},{z_position}\n"
    with lock:
        with open(metadata_file, "a") as f:
            f.write(metadata_entry)
        
        print(f"Samples completed: {i}/{num_samples}", end="\r")
        

def create_paired_synthetic_experiment(name, num_samples, atlas):
    output_path = Path("~/Desktop/synthetic_experiments/").expanduser()
    output_path.mkdir(exist_ok=True)
    experiment_path = output_path / name
    experiment_path.mkdir(exist_ok=True)

    metadata_file = experiment_path / "metadata.csv"

    # Open the metadata file in write mode to write the header
    with open(metadata_file, "w") as f:
        f.write("filename,x_angle,y_angle,z_position\n")

    lock = threading.Lock()
    ttrs = []
    with ThreadPoolExecutor(max_workers=128) as executor:
        futures = [
            executor.submit(
                generate_sample_pair,
                i,
                num_samples,
                atlas,
                experiment_path,
                metadata_file,
                ttrs,
                lock,
            )
            for i in range(num_samples)
        ]
        for future in as_completed(futures):
            future.result()


def create_synthetic_experiment(name, num_samples, atlas, annotation):
    output_path = Path("~/Desktop/synthetic_experiments/").expanduser()
    output_path.mkdir(exist_ok=True)
    experiment_path = output_path / name
    experiment_path.mkdir(exist_ok=True)

    metadata_file = experiment_path / "metadata.csv"

    # Open the metadata file in write mode to write the header
    with open(metadata_file, "w") as f:
        f.write("filename,x_angle,y_angle,z_position\n")

    lock = threading.Lock()
    ttrs = []
    with ThreadPoolExecutor(max_workers=128) as executor:
        futures = [
            executor.submit(
                generate_sample,
                i,
                num_samples,
                atlas,
                annotation,
                experiment_path,
                metadata_file,
                ttrs,
                lock,
            )
            for i in range(num_samples)
        ]
        for future in as_completed(futures):
            future.result()


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
        parents = ["567", "971", "940", "443", "1099", "579", "484682520", "484682512"]
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
    atlas_path = Path("~/.belljar/nrrd/atlas_10.nrrd")
    annotation_path = Path("~/.belljar/nrrd/annotation_10.nrrd")
    atlas, _ = nrrd.read(str(atlas_path.expanduser()))
    # annotation, annotation_header = nrrd.read(str(annotation_path.expanduser()))
    # create_synthetic_experiment("big_random_set_2", 10_000_000, atlas, annotation)
    create_paired_synthetic_experiment("big_random_set_paired", 100_000, atlas)

if __name__ == "__main__":
    main()
