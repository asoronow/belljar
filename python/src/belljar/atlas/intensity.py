"""Intensity normalisation for atlas reference volumes.

BrainGlobe references are stored as uint16 with stain-dependent ranges
(the Allen Nissl volume spans roughly [0, 27433]). Every consumer of the
reference works in uint8, so the volume is rescaled once, here, and both
training data generation and alignment see the same gray levels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def rescale_reference(
    volume: NDArray,
    sample_voxels: int = 4_000_000,
) -> NDArray[np.uint8]:
    """Map nonzero voxel intensities onto [0, 255] by their 1st/99th percentiles.

    Background (zero) voxels stay zero. The percentiles are estimated on a
    strided subsample of at most ``sample_voxels`` voxels so that full-size
    atlases (over a billion voxels) do not need to be materialised as floats.

    Args:
        volume: 3D reference volume of any integer or float dtype.
        sample_voxels: Upper bound on the number of voxels used to estimate
            the percentiles.

    Returns:
        A uint8 volume with the same shape as ``volume``.
    """
    volume = np.asarray(volume)
    stride = max(1, int(np.ceil((volume.size / sample_voxels) ** (1.0 / volume.ndim))))
    sample = volume[(slice(None, None, stride),) * volume.ndim]
    nonzero = sample[sample > 0]

    if nonzero.size == 0:
        return np.zeros(volume.shape, dtype=np.uint8)

    low, high = np.percentile(nonzero, [1, 99])
    if high <= low:
        return np.clip(volume, 0, 255).astype(np.uint8)

    scale = 255.0 / (high - low)
    out = np.empty(volume.shape, dtype=np.uint8)
    # One z-plane at a time bounds the float working set to a single slice.
    for z in range(volume.shape[0]):
        plane = (volume[z].astype(np.float32) - low) * scale
        out[z] = np.clip(plane, 0, 255).astype(np.uint8)
    return out
