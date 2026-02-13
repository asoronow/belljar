"""Image preprocessing for registration.

Extracted from demons.py — provides feature extraction and histogram
matching utilities used before running registration.
"""

from __future__ import annotations

import cv2
import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray
from skimage.filters import sobel


def preprocess_for_registration(image: sitk.Image) -> sitk.Image:
    """Preprocess an image for registration by extracting edge features.

    Applies Gaussian blur followed by Sobel edge detection to create
    a feature representation that is robust across different staining
    intensities.

    Args:
        image: Input SimpleITK image.

    Returns:
        Edge-enhanced SimpleITK image (float32, normalized to [0, 1]).
    """
    image_array = sitk.GetArrayFromImage(sitk.Cast(image, sitk.sitkUInt8))
    blurred = cv2.GaussianBlur(image_array, (5, 5), 0)
    edges = sobel(blurred)

    # Normalize to [0, 1]
    edge_min, edge_max = edges.min(), edges.max()
    if edge_max > edge_min:
        edges = (edges - edge_min) / (edge_max - edge_min)
    else:
        edges = np.zeros_like(edges)

    edges = edges.astype(np.float32)
    return sitk.GetImageFromArray(edges)


def match_histograms(
    to_match: sitk.Image,
    match_to: sitk.Image,
    num_levels: int = 1024,
    num_match_points: int = 10,
) -> sitk.Image:
    """Match the histogram of one image to another using SimpleITK.

    Args:
        to_match: Image whose histogram will be adjusted.
        match_to: Reference image to match to.
        num_levels: Number of histogram levels.
        num_match_points: Number of match points for the histogram.

    Returns:
        Histogram-matched image.
    """
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(num_levels)
    matcher.SetNumberOfMatchPoints(num_match_points)
    matcher.ThresholdAtMeanIntensityOn()
    return matcher.Execute(to_match, match_to)


def apply_layer_intensity_adjustments(
    section: NDArray,
    label: NDArray,
    structure_map: dict,
    adjustments: dict[str, int] | None = None,
) -> NDArray:
    """Apply layer-specific intensity adjustments to a tissue section.

    Vectorized replacement for the pixel-by-pixel loop in the original demons.py.

    Args:
        section: 2D tissue image (uint8).
        label: 2D annotation image (uint32 region IDs).
        structure_map: Dict mapping region_id -> {name, ...}.
        adjustments: Dict mapping layer name substring -> intensity delta.
                     Defaults to {"layer 4": 15, "layer 5": -7}.

    Returns:
        Adjusted section image (uint8).
    """
    if adjustments is None:
        adjustments = {"layer 4": 15, "layer 5": -7}

    result = section.astype(np.int16)
    label_flat = label.ravel()

    # Pre-compute masks for each adjustment layer
    for layer_name, delta in adjustments.items():
        layer_mask = np.zeros(label_flat.shape, dtype=bool)
        for region_id, info in structure_map.items():
            if layer_name in info["name"].lower():
                layer_mask |= label_flat == region_id
        result_flat = result.ravel()
        result_flat[layer_mask] = np.clip(result_flat[layer_mask] + delta, 0, 255)

    return result.astype(np.uint8)


def resize_nearest_neighbor(image: NDArray, new_size: tuple[int, int]) -> NDArray:
    """Resize an image using nearest-neighbor interpolation via SimpleITK.

    Used for annotation volumes where interpolation would create invalid labels.

    Args:
        image: Input numpy array.
        new_size: Target size as (width, height).

    Returns:
        Resized numpy array with original dtype preserved.
    """
    sitk_image = sitk.GetImageFromArray(image)
    original_size = sitk_image.GetSize()
    original_spacing = sitk_image.GetSpacing()
    new_spacing = [
        float(orig_space) * float(orig_size) / float(new_dim)
        for orig_space, orig_size, new_dim in zip(original_spacing, original_size, new_size)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputPixelType(sitk_image.GetPixelIDValue())
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(sitk_image.GetOrigin())
    resampler.SetOutputDirection(sitk_image.GetDirection())

    resized = resampler.Execute(sitk_image)
    return sitk.GetArrayFromImage(resized)
