import os
import pickle
import argparse
from pathlib import Path
import tifffile
import numpy as np
from demons import resize_image_nearest_neighbor
import cv2


def reconstruct_region(intensity_data):
    """
    Reconstruct a region from its intensity data.

    Args:
        intensity_data (dict): Dictionary of intensity data. Keys are points, values are pixel intensity.

    Returns:
        numpy.ndarray: 2D numpy array of reconstructed region.
    """

    # Get the max x and y values
    max_x = max([point[0] for point in intensity_data.keys()])
    max_y = max([point[1] for point in intensity_data.keys()])
    # Create a blank image
    blank = np.zeros((max_x + 1, max_y + 1))
    # Fill in the image
    for point, intensity in intensity_data.items():
        blank[point] = intensity

    return blank


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate the average intensity of a region in normalized coordinates"
    )

    parser.add_argument(
        "-i", "--images", help="input directory for intensity images", default=""
    )
    parser.add_argument(
        "-o", "--output", help="output directory for average intensity pkl", default=""
    )
    parser.add_argument(
        "-a", "--annotations", help="input directory for annotation pkls", default=""
    )
    parser.add_argument(
        "-m", "--map", help="input directory for structure map", default=""
    )
    parser.add_argument(
        "-w",
        "--whole",
        help="Set True to process a whole brain slice (Default is False)",
        default=False,
    )
    args = parser.parse_args()

    # Read in the intensity images
    intensityPath = args.images.strip()
    intensityFiles = os.listdir(intensityPath)
    intensityFiles.sort()
    is_whole = eval(args.whole.strip())
    # Read the annotation for the images
    annotationPath = args.annotations.strip()
    annotationFiles = os.listdir(annotationPath)
    annotationFiles = [f for f in annotationFiles if f.endswith(".pkl")]
    annotationFiles.sort()

    print(2 + len(intensityFiles), flush=True)
    print("Setting up...", flush=True)

    structure_map = pickle.load(open(args.map.strip(), "rb"))

    for i, iName in enumerate(intensityFiles):
        # load the image
        try:
            intensity = tifffile.imread(intensityPath + "/" + iName)
            # get the image width and height
            height, width = intensity.shape
        except:
            print(f"Erorr loading {iName}! Channels > 1 or bad image.", flush=True)
            continue

        # load the annotation
        with open(annotationPath + "/" + annotationFiles[i], "rb") as f:
            print("Processing " + iName, flush=True)
            annotation = pickle.load(f)

            annotation_recaled = resize_image_nearest_neighbor(
                annotation, (width, height)
            )
            required_regions = [
                "VISa",
                "VISal",
                "VISam",
                "VISp",
                "VISl",
                "VISli",
                "VISpl",
                "VISpm",
                "VISpor",
                "VISrl",
                "RSPagl",
                "RSPd",
                "RSPv",
            ]

            required_ids = [
                atlas_id
                for atlas_id, data in structure_map.items()
                if data["acronym"] in required_regions
            ]

            intensities = {required_id: {} for required_id in required_ids}

            # Get all children of the required regions in a dict
            # Dict helps us check which parent a child belongs to
            # Child == Parent in ID_PATH
            children_ids = {required_id: [] for required_id in required_ids}
            for required_id in required_ids:
                for atlas_id, data in structure_map.items():
                    if required_id in [
                        int(sub_id) for sub_id in data["id_path"].split("/")
                    ]:
                        children_ids[required_id].append(atlas_id)

            # Scan resized annotation for any child ids
            # If found, add its vertex and intensity to the parent
            for parent_id, children in children_ids.items():
                for child_id in children:
                    # Get the vertex of the child
                    verts = np.where(annotation_recaled == child_id)
                    if verts[0].size == 0:
                        continue
                    for point in zip(*verts):
                        # check if whole
                        if not is_whole:
                            intensities[parent_id][point] = intensity[point]
                        else:
                            # take only points in the left half
                            if point[1] < width // 2:
                                intensities[parent_id][point] = intensity[point]

            # Save the intensity values and the verticies as ROI package pkls
            for region in intensities.keys():
                # reconstruct the region
                if intensities[region] == {}:  # skip empty regions
                    continue

                name = iName.split(".")[0]
                region_name = structure_map[region]["acronym"]

                # debug = reconstruct_region(intensities[region])
                # debug = cv2.normalize(debug, None, 0, 255, cv2.NORM_MINMAX)
                # cv2.imwrite(
                #     str(
                #         Path(
                #             args.output.strip()
                #             + "/"
                #             + f"{name}_{region_name}_debug"
                #             + ".png"
                #         )
                #     ),
                #     debug,
                # )
                # split file name
                outputPath = Path(
                    args.output.strip() + "/" + f"{name}_{region_name}" + ".pkl"
                )
                with open(outputPath, "wb") as f:
                    pickle.dump(
                        {
                            "roi": intensities[region],
                            "name": region_name,
                        },
                        f,
                    )

    print("Done!", flush=True)
