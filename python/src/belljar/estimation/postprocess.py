"""Post-processing for slice position estimates.

Implements angle integration and cutting index weighting to regularize
predictions across serial sections. Following the DeepSlice approach,
consecutive sections from the same brain share the same cutting angles,
so averaging/smoothing angles across sections improves accuracy.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def integrate_angles(
    predictions: list[dict[str, Any]],
    window_size: int = 5,
) -> list[dict[str, Any]]:
    """Smooth cutting angles across serial sections.

    Consecutive sections from the same cutting session share the same
    cutting angle. This function applies a rolling median filter to
    the predicted angles to remove outliers while preserving the true
    cutting direction.

    Args:
        predictions: List of prediction dicts (must have 'x_angle' and 'y_angle').
        window_size: Size of the rolling median window. Must be odd.

    Returns:
        Updated predictions with smoothed angles.
    """
    if len(predictions) < 3:
        return predictions

    if window_size % 2 == 0:
        window_size += 1

    x_angles = np.array([p["x_angle"] for p in predictions])
    y_angles = np.array([p["y_angle"] for p in predictions])

    x_smoothed = _rolling_median(x_angles, window_size)
    y_smoothed = _rolling_median(y_angles, window_size)

    result = []
    for i, pred in enumerate(predictions):
        updated = dict(pred)
        updated["x_angle_raw"] = updated["x_angle"]
        updated["y_angle_raw"] = updated["y_angle"]
        updated["x_angle"] = float(x_smoothed[i])
        updated["y_angle"] = float(y_smoothed[i])
        result.append(updated)

    return result


def _rolling_median(values: NDArray, window: int) -> NDArray:
    """Apply a rolling median filter with edge padding."""
    n = len(values)
    half = window // 2
    padded = np.pad(values, half, mode="edge")
    result = np.empty(n)
    for i in range(n):
        result[i] = np.median(padded[i : i + window])
    return result


def regularize_spacing(
    predictions: list[dict[str, Any]],
    section_spacing_um: float = 50.0,
    atlas_resolution_um: float = 10.0,
) -> list[dict[str, Any]]:
    """Regularize AP positions based on expected section spacing.

    For uniformly-cut serial sections, the AP positions should be
    approximately evenly spaced. This function adjusts positions to
    follow a smooth monotonic trajectory.

    Args:
        predictions: List of prediction dicts (must have 'z_position').
        section_spacing_um: Physical spacing between consecutive sections in micrometers.
        atlas_resolution_um: Atlas voxel size in micrometers.

    Returns:
        Updated predictions with regularized z_positions.
    """
    if len(predictions) < 2:
        return predictions

    expected_step = section_spacing_um / atlas_resolution_um

    z_positions = np.array([p["z_position"] for p in predictions])

    # Determine direction (ascending or descending)
    direction = np.sign(z_positions[-1] - z_positions[0])
    if direction == 0:
        direction = 1.0

    # Fit a linear model to the predicted positions
    indices = np.arange(len(z_positions))
    coeffs = np.polyfit(indices, z_positions, deg=1)
    fitted = np.polyval(coeffs, indices)

    # Blend between raw predictions and linear fit
    # Weight toward the linear fit to enforce spacing regularity
    alpha = 0.5  # blending weight (0 = all raw, 1 = all linear)
    regularized = (1 - alpha) * z_positions + alpha * fitted

    result = []
    for i, pred in enumerate(predictions):
        updated = dict(pred)
        updated["z_position_raw"] = updated["z_position"]
        updated["z_position"] = float(regularized[i])
        result.append(updated)

    return result


def denormalize_predictions(
    predictions: list[dict[str, Any]],
    ap_range: tuple[float, float] = (0.0, 1324.0),
    angle_range: tuple[float, float] = (-10.0, 10.0),
) -> list[dict[str, Any]]:
    """Convert normalized model outputs to physical values.

    Args:
        predictions: List of prediction dicts with normalized values.
        ap_range: (min, max) for AP position.
        angle_range: (min, max) for angles.

    Returns:
        Predictions with denormalized z_position, x_angle, y_angle.
    """
    result = []
    for pred in predictions:
        updated = dict(pred)
        if "z_position" in updated:
            z = updated["z_position"]
            updated["z_position"] = z * (ap_range[1] - ap_range[0]) + ap_range[0]
        if "x_angle" in updated:
            a = updated["x_angle"]
            updated["x_angle"] = a * (angle_range[1] - angle_range[0]) + angle_range[0]
        if "y_angle" in updated:
            a = updated["y_angle"]
            updated["y_angle"] = a * (angle_range[1] - angle_range[0]) + angle_range[0]
        result.append(updated)
    return result
