"""3D atlas volume slicing with full rotation support.

Replaces the simple linear tilt model in the original slice_atlas.py with
proper 3D rotation matrices via scipy.spatial.transform.Rotation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import map_coordinates
from scipy.spatial.transform import Rotation


def slice_3d_volume(
    volume: NDArray,
    z_position: float,
    x_angle: float = 0.0,
    y_angle: float = 0.0,
    z_angle: float = 0.0,
    order: int = 1,
) -> NDArray:
    """Extract an oblique 2D slice from a 3D volume using full 3D rotation.

    Args:
        volume: 3D numpy array with shape (Z, Y, X).
        z_position: Position along the z-axis (AP position) for the slice center.
        x_angle: Tilt around the X axis in degrees.
        y_angle: Tilt around the Y axis in degrees.
        z_angle: In-plane rotation in degrees.
        order: Interpolation order. Use 0 for annotation volumes (nearest-neighbor
               preserves label integrity) and 1 for atlas images (linear interpolation
               eliminates aliasing).

    Returns:
        2D numpy array of the extracted slice.
    """
    z_dim, y_dim, x_dim = volume.shape
    center = np.array([z_position, y_dim / 2.0, x_dim / 2.0])

    # Create 2D coordinate grid centered at origin
    y_coords, x_coords = np.meshgrid(
        np.arange(y_dim, dtype=np.float64) - y_dim / 2.0,
        np.arange(x_dim, dtype=np.float64) - x_dim / 2.0,
        indexing="ij",
    )
    z_coords = np.zeros_like(y_coords)

    # Stack into (N, 3) array of [z, y, x] points
    points = np.stack(
        [z_coords.ravel(), y_coords.ravel(), x_coords.ravel()],
        axis=1,
    )

    # Apply 3D rotation
    rot = Rotation.from_euler("xyz", [x_angle, y_angle, z_angle], degrees=True)
    rotated_points = rot.apply(points)

    # Translate to volume coordinates
    sampling_coords = rotated_points + center

    # Clamp to volume bounds to avoid edge artifacts
    for dim_idx, dim_size in enumerate(volume.shape):
        sampling_coords[:, dim_idx] = np.clip(
            sampling_coords[:, dim_idx], 0, dim_size - 1
        )

    # Sample from volume
    slice_2d = map_coordinates(
        volume,
        sampling_coords.T,
        order=order,
        mode="constant",
        cval=0,
    ).reshape(y_dim, x_dim)

    return slice_2d


def slice_atlas_and_annotation(
    atlas: NDArray,
    annotation: NDArray,
    z_position: float,
    x_angle: float = 0.0,
    y_angle: float = 0.0,
    z_angle: float = 0.0,
) -> tuple[NDArray, NDArray]:
    """Slice both atlas and annotation volumes at the same position/angles.

    Uses linear interpolation for the atlas image and nearest-neighbor
    for the annotation volume to preserve label integrity.

    Args:
        atlas: 3D atlas reference volume (uint8).
        annotation: 3D annotation volume (uint32 region IDs).
        z_position: AP position.
        x_angle: X tilt in degrees.
        y_angle: Y tilt in degrees.
        z_angle: In-plane rotation in degrees.

    Returns:
        Tuple of (atlas_slice, annotation_slice).
    """
    atlas_slice = slice_3d_volume(
        atlas, z_position, x_angle, y_angle, z_angle, order=1
    ).astype(np.uint8)

    annotation_slice = slice_3d_volume(
        annotation, z_position, x_angle, y_angle, z_angle, order=0
    ).astype(np.uint32)

    return atlas_slice, annotation_slice
