"""Image format handling utilities.

Centralized image I/O supporting TIFF (8/16-bit), PNG, JPG, and
multi-channel formats.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff
from numpy.typing import NDArray

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def is_supported_image(path: Path) -> bool:
    """Check if a file has a supported image extension."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def list_images(directory: Path) -> list[Path]:
    """List all supported image files in a directory, sorted by name.

    Args:
        directory: Directory to scan.

    Returns:
        Sorted list of image file paths.
    """
    files = [
        f for f in directory.iterdir()
        if f.is_file() and not f.name.startswith(".") and is_supported_image(f)
    ]
    files.sort(key=lambda f: f.name)
    return files


def read_image(path: Path, grayscale: bool = False) -> NDArray:
    """Read an image file, handling TIFF and standard formats.

    For 16-bit TIFFs, automatically converts to uint8.

    Args:
        path: Path to the image file.
        grayscale: If True, convert to grayscale.

    Returns:
        Image as numpy array (uint8).
    """
    ext = path.suffix.lower()

    if ext in (".tif", ".tiff"):
        img = tiff.imread(str(path))
        # Handle 16-bit
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        elif img.dtype in (np.float32, np.float64):
            img = (img * 255).astype(np.uint8)
    else:
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(str(path), flag)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")

    if grayscale and len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img
