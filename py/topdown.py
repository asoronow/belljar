import nrrd
import cv2
from pathlib import Path
import numpy as np
import pickle
from scipy.ndimage import rotate


def main():
    """Load in the atlas and get the dorsal down outline of all HVAs"""
    regions_to_map = [
        "VISp",
        "VISl",
        "VISpm",
        "VISam",
        "VISrl",
        "VISal",
        "VISa",
        "VISli",
        "RSPv",
        "TEa",
        "VISpor",
    ]

    class_map_path = Path("/Users/alec/Projects/belljar/csv/class_map.pkl")
    class_map = None
    with open(class_map_path, "rb") as cm:
        class_map = pickle.load(cm)

    print("Mapping...")
    regions_as_ids = []
    to_merge = {}
    for region_id, region_data in class_map.items():
        for i, required_region in enumerate(regions_to_map):
            if "layer" in region_data["name"].lower():
                layer_endings = ["1", "2/3", "4", "5", "6a", "6b"]
                # replace the layer endings with nothing
                acronym = region_data["acronym"]
                for ending in layer_endings:
                    acronym = acronym.replace(ending, "")
                if required_region.lower() == acronym.lower():
                    parent_id = np.int32(class_map[region_id]["id_path"].split("/")[-3])
                    to_merge[parent_id] = {
                        "color": np.random.randint(0, 255, size=3),
                        "acronym": required_region,
                    }

                    regions_as_ids.append(region_id)
    # merge dicts
    class_map = {**class_map, **to_merge}

    atlas_path = Path("~/.belljar/nrrd/annotation_10_all.nrrd")
    annotation, _ = nrrd.read(atlas_path.expanduser(), index_order="C")
    print("Rotating...")
    annotation = rotate(annotation, -20, axes=(1, 2), reshape=True, order=0)
    annotation = rotate(annotation, -10, axes=(0, 1), reshape=True, order=0)
    z, y, x = annotation.shape

    # slice in z-x plane
    print("Drawing...")
    dorsal_slice_colored = np.zeros((z, x, 3), dtype=np.uint8)
    dorsal_slice_maps = {}
    for i in range(y):
        dorsal_slice = annotation[:, i, :]
        for region_id in np.unique(dorsal_slice):
            # where that region is in the slice
            # add its color from the class map to the colored slice
            if region_id in class_map and region_id in regions_as_ids:
                # add any points here to the mask
                if "layer" in class_map[region_id]["name"].lower():
                    parent_id = np.int32(class_map[region_id]["id_path"].split("/")[-3])
                    if parent_id not in dorsal_slice_maps:
                        dorsal_slice_maps[parent_id] = np.zeros((z, x), dtype=np.uint8)

                    dorsal_slice_maps[parent_id][dorsal_slice == region_id] = 255

    # Remove any intersecting masks
    # Making sure RSPv is on the bottom
    for mapping in sorted(dorsal_slice_maps, reverse=True):
        for other_mapping in dorsal_slice_maps:
            if mapping != other_mapping:
                dorsal_slice_maps[mapping][dorsal_slice_maps[other_mapping] == 255] = 0

    # Draw the outlines on the colored slice
    for mapping in dorsal_slice_maps:
        # find contour
        contours, _ = cv2.findContours(
            dorsal_slice_maps[mapping], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        # draw contour
        cv2.drawContours(
            dorsal_slice_colored,
            contours,
            -1,
            (255, 255, 255),
            2,
        )

    cv2.imwrite("dorsal_slice_colored.png", dorsal_slice_colored)
    for mapping in dorsal_slice_maps:
        # center
        c_x = int(np.mean(np.where(dorsal_slice_maps[mapping] == 255)[1]))
        c_y = int(np.mean(np.where(dorsal_slice_maps[mapping] == 255)[0]))
        # draw a breakout line from the center to the label
        cv2.putText(
            dorsal_slice_colored,
            class_map[mapping]["acronym"],
            (c_x - 20, c_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imshow(f"rotation", dorsal_slice_colored)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
