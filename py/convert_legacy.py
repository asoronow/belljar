import nrrd
import numpy as np
from pathlib import Path
import json
import pickle
import cv2
from slice_atlas import add_outlines, slice_3d_volume


def main():
    # load in the legacy graph
    graph_path = Path("./csv/legacy_graph.json")
    current_map = Path("./csv/structure_map.pkl")
    # legacy atlas
    atlas_path = Path("C:/Users/Alec/.belljar/nrrd/reconstructed_atlas.nrrd")
    atlas = nrrd.read(atlas_path)[0]

    # legacy annotations
    annotation_path = Path("C:/Users/Alec/.belljar/nrrd/reconstructed_annotation.nrrd")
    annotation = nrrd.read(annotation_path)[0]

    # load in the current map
    acronym_to_id = {}
    with open(current_map, "rb") as f:
        current_map = pickle.load(f)
        for id, info in current_map.items():
            acronym_to_id[info["acronym"]] = id

    # with open(graph_path, "r") as f:
    #     graph = json.load(f)

    # correspondences = 0
    # mismatches = 0
    # reconstructed = np.zeros_like(annotation, dtype=np.uint32)
    # for i, region in enumerate(graph):
    #     try:
    #         if region in acronym_to_id.keys():
    #             print(f"Found {region} has corresponding id {acronym_to_id[region]}")
    #             reconstructed[annotation == i] = acronym_to_id[region]
    #             correspondences += 1
    #         else:
    #             print(f"No corresponding id found for {region}")
    #             mismatches += 1
    #     except:
    #         mismatches += 1
    # print(f"Found {correspondences} matches and {mismatches} mismatches")

    # # save the reconstructed annotation
    annotation = annotation.astype(np.uint32)
    print(annotation.shape)
    midway_slice = slice_3d_volume(annotation, annotation.shape[0] // 2, 0, 10)
    midway_atlas = slice_3d_volume(atlas, atlas.shape[0] // 2, 0, 10)
    colored = np.zeros(midway_slice.shape + (3,), dtype=np.uint8)
    for label in np.unique(midway_slice):
        colored[midway_slice == label] = current_map[label]["color"]

    # rotate 90 degrees right
    colored = add_outlines(midway_slice, colored)
    cv2.imshow("colored", colored)
    cv2.imshow("atlas", midway_atlas)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
