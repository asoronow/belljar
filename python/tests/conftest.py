"""Shared test fixtures for belljar tests."""

import numpy as np
import pytest


@pytest.fixture
def synthetic_atlas() -> np.ndarray:
    """Create a small synthetic atlas volume for testing.

    Returns a 100x60x50 uint8 volume with distinct regions.
    """
    volume = np.zeros((100, 60, 50), dtype=np.uint8)
    # Create some structure: gradient along z-axis
    for z in range(100):
        volume[z] = int(255 * z / 99)
    # Add a bright sphere in the center
    z, y, x = np.ogrid[30:70, 15:45, 10:40]
    sphere_mask = ((z - 50) ** 2 + (y - 30) ** 2 + (x - 25) ** 2) < 15**2
    volume[30:70, 15:45, 10:40][sphere_mask] = 200
    return volume


@pytest.fixture
def synthetic_annotation() -> np.ndarray:
    """Create a small synthetic annotation volume for testing.

    Returns a 100x60x50 uint32 volume with labeled regions.
    """
    annotation = np.zeros((100, 60, 50), dtype=np.uint32)
    # Region 1: top half
    annotation[:50] = 1
    # Region 2: bottom half
    annotation[50:] = 2
    # Region 3: center sphere
    z, y, x = np.ogrid[30:70, 15:45, 10:40]
    sphere_mask = ((z - 50) ** 2 + (y - 30) ** 2 + (x - 25) ** 2) < 15**2
    annotation[30:70, 15:45, 10:40][sphere_mask] = 3
    return annotation


def _structured_slice(size: int) -> np.ndarray:
    """Blurred discs on a two-tone background: enough structure to register."""
    import cv2

    yy, xx = np.mgrid[0:size, 0:size]
    image = np.full((size, size), 40.0, dtype=np.float32)
    image[: size // 2] = 90.0
    scale = size / 64
    for cx, cy, radius, value in [
        (32, 32, 16, 220),
        (22, 30, 4, 60),
        (42, 36, 3, 160),
        (32, 18, 3, 250),
    ]:
        mask = ((xx - cx * scale) ** 2 + (yy - cy * scale) ** 2) < (radius * scale) ** 2
        image[mask] = value
    blurred = cv2.GaussianBlur(image, (0, 0), 1.5 * scale)
    return blurred.astype(np.uint8)


@pytest.fixture
def structured_tissue() -> np.ndarray:
    """256x256 tissue image that matches ``structured_provider`` slices."""
    return _structured_slice(256)


@pytest.fixture
def structured_provider():
    """AtlasProvider over a 100x64x64 volume whose every z slice has the same
    structure: region 1 top half, region 2 bottom half, region 3 central disc."""
    from unittest.mock import MagicMock

    from belljar.atlas.provider import AtlasProvider

    size = 64
    reference = np.repeat(_structured_slice(size)[None], 100, axis=0)
    annotation = np.zeros((100, size, size), dtype=np.uint32)
    annotation[:, : size // 2] = 1
    annotation[:, size // 2 :] = 2
    yy, xx = np.mgrid[0:size, 0:size]
    disc = ((xx - 32) ** 2 + (yy - 32) ** 2) < 16**2
    annotation[:, disc] = 3

    atlas = MagicMock()
    atlas.reference = reference
    atlas.annotation = annotation
    atlas.resolution = (10.0, 10.0, 10.0)
    atlas.structures = {
        1: {"name": "Top", "acronym": "TOP", "rgb_triplet": [255, 0, 0]},
        2: {"name": "Bottom", "acronym": "BOT", "rgb_triplet": [0, 255, 0]},
        3: {"name": "Disc", "acronym": "DSC", "rgb_triplet": [0, 0, 255]},
    }
    for region_id, structure in atlas.structures.items():
        structure["structure_id_path"] = [997, region_id]
    provider = AtlasProvider("test_atlas")
    provider._atlas = atlas
    return provider
