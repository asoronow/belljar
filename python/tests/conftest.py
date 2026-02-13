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
