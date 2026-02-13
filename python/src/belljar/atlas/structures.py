"""Brain region hierarchy, masking, and visualization utilities.

Extracted from slice_atlas.py — handles structure maps, region masking,
and annotation outline drawing.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from skimage import measure


def mask_slice_by_region(
    atlas_slice: NDArray,
    annotation_slice: NDArray,
    structure_map: dict,
    region: str,
    cerebrum_parent_ids: list[str] | None = None,
) -> tuple[NDArray, NDArray]:
    """Mask a slice to include only cerebrum or non-cerebrum regions.

    Args:
        atlas_slice: 2D atlas image (uint8).
        annotation_slice: 2D annotation image (uint32 region IDs).
        structure_map: Dict mapping region_id -> {name, acronym, color, id_path}.
        region: "C" for cerebrum only, "NC" for non-cerebrum.
        cerebrum_parent_ids: Parent IDs defining the cerebrum. Uses defaults if None.

    Returns:
        Tuple of (masked_atlas, masked_annotation).
    """
    if cerebrum_parent_ids is None:
        cerebrum_parent_ids = [
            "567", "971", "940", "443", "1099", "579", "484682520", "484682512",
        ]

    # Build set of cerebrum region IDs
    cerebrum_ids = set()
    non_cerebrum_ids = set()
    for region_id, info in structure_map.items():
        id_parts = info["id_path"].split("/")
        if any(parent in id_parts for parent in cerebrum_parent_ids):
            cerebrum_ids.add(region_id)
        else:
            non_cerebrum_ids.add(region_id)

    # Select which IDs to keep
    keep_ids = cerebrum_ids if region == "C" else non_cerebrum_ids

    # Vectorized masking
    keep_mask = np.isin(annotation_slice, list(keep_ids))
    masked_atlas = np.where(keep_mask, atlas_slice, 0).astype(np.uint8)
    masked_annotation = np.where(keep_mask, annotation_slice, 0).astype(np.uint32)

    return masked_atlas, masked_annotation


def add_outlines(
    annotation_slice: NDArray,
    color_annotation: NDArray,
) -> NDArray:
    """Add white outlines between unique regions in the annotation.

    Args:
        annotation_slice: 2D annotation image (uint32 region IDs).
        color_annotation: 2D RGB color image to draw outlines on.

    Returns:
        Color image with region boundary outlines drawn in black.
    """
    result = color_annotation.copy()
    unique_ids = np.unique(annotation_slice)

    for region_id in unique_ids:
        if region_id == 0:
            continue
        contours = measure.find_contours(annotation_slice == region_id, 0.5)
        for contour in contours:
            rows = np.clip(contour[:, 0].astype(np.int32), 0, result.shape[0] - 1)
            cols = np.clip(contour[:, 1].astype(np.int32), 0, result.shape[1] - 1)
            result[rows, cols] = 0

    return result


def colorize_annotation(
    annotation_slice: NDArray,
    structure_map: dict,
) -> NDArray:
    """Convert an annotation slice to an RGB color image using the structure map.

    Args:
        annotation_slice: 2D annotation image (uint32 region IDs).
        structure_map: Dict mapping region_id -> {name, acronym, color, id_path}.

    Returns:
        RGB color image (uint8, shape HxWx3).
    """
    color_image = np.zeros((*annotation_slice.shape, 3), dtype=np.uint8)

    for region_id, info in structure_map.items():
        mask = annotation_slice == region_id
        if np.any(mask):
            color_image[mask] = info["color"]

    return color_image
