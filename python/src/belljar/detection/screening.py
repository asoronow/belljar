"""Post-detection screening and filtering.

Extracted from find_neurons.py — provides configurable area-based,
statistical, and eccentricity-based filtering of detections.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops

logger = logging.getLogger(__name__)


def xyxy_to_area(box: list[float]) -> float:
    """Compute bounding box area from [x1, y1, x2, y2] format."""
    return (box[2] - box[0]) * (box[3] - box[1])


def check_eccentricity(
    box: list[float],
    threshold: float,
    image: NDArray,
) -> bool:
    """Check if the object in a bounding box exceeds an eccentricity threshold.

    Segments the cell using Otsu thresholding and computes the eccentricity
    of the largest region.

    Args:
        box: Bounding box [x1, y1, x2, y2].
        threshold: Eccentricity threshold (objects above this are removed).
        image: Full image (BGR uint8).

    Returns:
        True if the object exceeds the threshold (should be removed).
    """
    try:
        x1, y1, x2, y2 = [int(b) for b in box]
        pad = 5
        cell_image = image[
            max(y1 - pad, 0) : y2 + pad,
            max(x1 - pad, 0) : x2 + pad,
        ]
        if cell_image.size == 0:
            return False

        if len(cell_image.shape) > 2:
            cell_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)

        thresh = threshold_otsu(cell_image)
        mask = cell_image > thresh
        labeled_mask = label(mask)
        regions = regionprops(labeled_mask)

        if not regions:
            return False

        largest_region = max(regions, key=lambda r: r.area)
        return largest_region.eccentricity > threshold

    except Exception as e:
        logger.debug("Failed eccentricity check: %s", e)
        return True


def screen_predictions(
    prediction_objects: list[Any],
    area_threshold: float,
    eccentricity_threshold: float | None = None,
    image: NDArray | None = None,
) -> list[Any]:
    """Screen SAHI prediction objects by area and eccentricity.

    Two-pass filtering:
    1. Remove objects below the area threshold.
    2. Remove statistical outliers (> mean + 2*std area).
    3. Optionally filter by eccentricity.

    Args:
        prediction_objects: List of SAHI ObjectPrediction objects.
        area_threshold: Minimum bounding box area in pixels.
        eccentricity_threshold: Maximum eccentricity. None to skip.
        image: Full image required for eccentricity filtering.

    Returns:
        Filtered list of prediction objects.
    """
    # Pass 1: Area threshold
    first_pass = [
        obj for obj in prediction_objects
        if xyxy_to_area(obj.bbox.to_xyxy()) > area_threshold
    ]

    if len(first_pass) < 3:
        return first_pass

    # Pass 2: Statistical outlier removal
    areas = [xyxy_to_area(obj.bbox.to_xyxy()) for obj in first_pass]
    avg_area = np.mean(areas)
    std_area = np.std(areas)
    upper_bound = avg_area + 2 * std_area

    second_pass = [
        obj for obj in first_pass
        if xyxy_to_area(obj.bbox.to_xyxy()) < upper_bound
    ]

    # Pass 3: Eccentricity filtering (optional)
    if eccentricity_threshold is not None and image is not None:
        second_pass = [
            obj for obj in second_pass
            if not check_eccentricity(obj.bbox.to_xyxy(), eccentricity_threshold, image)
        ]
    elif eccentricity_threshold is not None and image is None:
        logger.warning("Image not provided — eccentricity screening skipped.")

    return second_pass
