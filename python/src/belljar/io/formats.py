"""Image format handling utilities.

Centralized image I/O supporting TIFF (8/16-bit), PNG, JPG, and
multi-channel formats.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
import tifffile as tiff

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    from numpy.typing import NDArray

TIFF_EXTENSIONS = {".tif", ".tiff"}
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", *TIFF_EXTENSIONS}


def is_supported_image(path: Path) -> bool:
    """Check if a file has a supported image extension."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def is_tiff(path: Path) -> bool:
    """True for any spelling of a .tif/.tiff file name, e.g. foo.TIFF."""
    return path.suffix.lower() in TIFF_EXTENSIONS


def section_stem(path: Path) -> str:
    """The file name up to its first dot: the key every pipeline step names its output by."""
    return path.name.split(".")[0]


def _list_matching(directory: Path, predicate: Callable[[Path], bool]) -> list[Path]:
    files = [
        f for f in directory.iterdir()
        if f.is_file() and not f.name.startswith(".") and predicate(f)
    ]
    files.sort(key=lambda f: f.name)
    return files


def list_images(directory: Path) -> list[Path]:
    """List all supported image files in a directory, sorted by name.

    Args:
        directory: Directory to scan.

    Returns:
        Sorted list of image file paths.
    """
    return _list_matching(directory, is_supported_image)


def list_tiffs(directory: Path) -> list[Path]:
    """List the TIFF files in a directory, sorted by name."""
    return _list_matching(directory, is_tiff)


def check_unique_stems(paths: Iterable[Path]) -> None:
    """Raise ValueError when two inputs share a section stem.

    Both would be written to the same <stem>.tif and the second would silently
    replace the first.
    """
    by_stem: dict[str, list[str]] = {}
    for path in paths:
        by_stem.setdefault(section_stem(path), []).append(path.name)
    collisions = {stem: names for stem, names in by_stem.items() if len(names) > 1}
    if collisions:
        lines = ["Input files would overwrite each other's output:"]
        for stem in sorted(collisions):
            lines.append(f"  {stem}.tif <- " + ", ".join(collisions[stem]))
        raise ValueError("\n".join(lines))


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
