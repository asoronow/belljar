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
    uncertainties: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Smooth cutting angles across serial sections.

    Consecutive sections from the same cutting session share the same
    cutting angle. This function applies a rolling median filter to
    the predicted angles to remove outliers while preserving the true
    cutting direction.

    When *uncertainties* are provided (one per prediction), an inverse-variance
    weighted rolling mean is used instead of the rolling median, so that
    high-uncertainty predictions are pulled more strongly toward their
    neighbors.

    Args:
        predictions: List of prediction dicts (must have 'x_angle' and 'y_angle').
        window_size: Size of the rolling median window. Must be odd.
        uncertainties: Per-section uncertainty values (higher = less certain).
            When ``None``, the original rolling median is used.

    Returns:
        Updated predictions with smoothed angles.
    """
    if len(predictions) < 3:
        return predictions

    if window_size % 2 == 0:
        window_size += 1

    x_angles = np.array([p["x_angle"] for p in predictions])
    y_angles = np.array([p["y_angle"] for p in predictions])

    if uncertainties is not None:
        weights = 1.0 / (np.array(uncertainties, dtype=np.float64) ** 2 + 1e-8)
        x_smoothed = _weighted_rolling_mean(x_angles, weights, window_size)
        y_smoothed = _weighted_rolling_mean(y_angles, weights, window_size)
    else:
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


def _weighted_rolling_mean(
    values: NDArray, weights: NDArray, window: int
) -> NDArray:
    """Inverse-variance weighted rolling mean with edge padding.

    Each value in the window is weighted by its corresponding weight
    (typically ``1 / (sigma^2 + eps)``).  Low-confidence (high uncertainty)
    sections therefore contribute less to the local estimate.
    """
    n = len(values)
    half = window // 2
    padded_vals = np.pad(values, half, mode="edge")
    padded_wts = np.pad(weights, half, mode="edge")
    result = np.empty(n)
    for i in range(n):
        w = padded_wts[i : i + window]
        v = padded_vals[i : i + window]
        result[i] = np.average(v, weights=w)
    return result


def regularize_spacing(
    predictions: list[dict[str, Any]],
    section_spacing_um: float = 50.0,
    atlas_resolution_um: float = 10.0,
    method: str = "linear",
) -> list[dict[str, Any]]:
    """Regularize AP positions based on expected section spacing.

    For uniformly-cut serial sections, the AP positions should be
    approximately evenly spaced. This function adjusts positions to
    follow a smooth monotonic trajectory.

    Args:
        predictions: List of prediction dicts (must have 'z_position').
        section_spacing_um: Physical spacing between consecutive sections in micrometers.
        atlas_resolution_um: Atlas voxel size in micrometers.
        method: Fitting method — ``"linear"`` (np.polyfit blend) or
            ``"ransac"`` (robust RANSAC fit that rejects outliers).

    Returns:
        Updated predictions with regularized z_positions.
    """
    if len(predictions) < 2:
        return predictions

    if method == "ransac":
        return regularize_spacing_ransac(
            predictions, section_spacing_um, atlas_resolution_um
        )

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


def regularize_spacing_ransac(
    predictions: list[dict[str, Any]],
    section_spacing_um: float = 50.0,
    atlas_resolution_um: float = 10.0,
) -> list[dict[str, Any]]:
    """Regularize AP positions using RANSAC for robust outlier rejection.

    Unlike the linear blend in :func:`regularize_spacing`, this fits a
    robust linear model via ``RANSACRegressor``.  Inliers are blended
    (alpha=0.5) with the robust fit while outliers are fully replaced by
    the fitted value.

    Args:
        predictions: List of prediction dicts (must have 'z_position').
        section_spacing_um: Physical spacing between consecutive sections in micrometers.
        atlas_resolution_um: Atlas voxel size in micrometers.

    Returns:
        Updated predictions with regularized z_positions.
    """
    from sklearn.linear_model import RANSACRegressor

    if len(predictions) < 2:
        return predictions

    expected_step = section_spacing_um / atlas_resolution_um

    z_positions = np.array([p["z_position"] for p in predictions])
    indices = np.arange(len(z_positions)).reshape(-1, 1)

    ransac = RANSACRegressor(
        residual_threshold=expected_step * 2,
        random_state=0,
    )
    ransac.fit(indices, z_positions)
    fitted = ransac.predict(indices)
    inlier_mask = ransac.inlier_mask_

    alpha = 0.5
    regularized = np.empty_like(z_positions)
    for i in range(len(z_positions)):
        if inlier_mask[i]:
            regularized[i] = (1 - alpha) * z_positions[i] + alpha * fitted[i]
        else:
            # Outliers get fully replaced by the robust fit
            regularized[i] = fitted[i]

    result = []
    for i, pred in enumerate(predictions):
        updated = dict(pred)
        updated["z_position_raw"] = updated["z_position"]
        updated["z_position"] = float(regularized[i])
        result.append(updated)

    return result


def enforce_orthogonality(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Enforce orthogonality and unit length on anchoring direction vectors.

    For each prediction that contains an ``anchoring_vectors`` key (dict
    with ox, oy, oz, ux, uy, uz, vx, vy, vz), applies numpy Gram-Schmidt
    orthogonalization to the u and v direction vectors so that:

    - u and v are unit length
    - u and v are orthogonal (u . v = 0)

    The origin (o) is left unchanged.  Predictions without an
    ``anchoring_vectors`` key are passed through unmodified.

    Also accepts an ``anchoring`` key containing a flat list/array of 9
    floats as a fallback format.

    Args:
        predictions: List of prediction dicts.

    Returns:
        Updated predictions with orthonormalized direction vectors.
    """
    result = []
    for pred in predictions:
        updated = dict(pred)

        av = updated.get("anchoring_vectors")
        flat_av = updated.get("anchoring")

        if av is not None and isinstance(av, dict):
            u = np.array([av["ux"], av["uy"], av["uz"]], dtype=np.float64)
            v = np.array([av["vx"], av["vy"], av["vz"]], dtype=np.float64)

            u_hat, v_hat = _gram_schmidt_np(u, v)

            updated["anchoring_vectors"] = dict(av)
            updated["anchoring_vectors"]["ux"] = float(u_hat[0])
            updated["anchoring_vectors"]["uy"] = float(u_hat[1])
            updated["anchoring_vectors"]["uz"] = float(u_hat[2])
            updated["anchoring_vectors"]["vx"] = float(v_hat[0])
            updated["anchoring_vectors"]["vy"] = float(v_hat[1])
            updated["anchoring_vectors"]["vz"] = float(v_hat[2])

        elif flat_av is not None and len(flat_av) == 9:
            a = np.array(flat_av, dtype=np.float64)
            o = a[0:3]
            u = a[3:6]
            v = a[6:9]

            u_hat, v_hat = _gram_schmidt_np(u, v)
            updated["anchoring"] = np.concatenate([o, u_hat, v_hat]).tolist()

        result.append(updated)
    return result


def _gram_schmidt_np(
    u: NDArray, v: NDArray
) -> tuple[NDArray, NDArray]:
    """Gram-Schmidt orthonormalization for two 3D vectors (numpy)."""
    u_norm = np.linalg.norm(u)
    u_hat = u / u_norm if u_norm > 1e-8 else u

    v_proj = np.dot(v, u_hat) * u_hat
    v_orth = v - v_proj
    v_norm = np.linalg.norm(v_orth)
    v_hat = v_orth / v_norm if v_norm > 1e-8 else v_orth

    return u_hat, v_hat


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
