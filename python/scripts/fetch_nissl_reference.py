#!/usr/bin/env python3
"""Download the Allen Mouse Brain Nissl volume and install it as a BrainGlobe additional reference.

Downloads ara_nissl_10.nrrd from the Allen Institute, converts to TIFF,
places it in the BrainGlobe atlas directory, and updates metadata.json.

Usage:
    python scripts/fetch_nissl_reference.py
    python scripts/fetch_nissl_reference.py --atlas-dir ~/.brainglobe/allen_mouse_10um_v1.2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

logger = logging.getLogger("belljar.fetch_nissl")

NISSL_URL = (
    "https://download.alleninstitute.org/informatics-archive/"
    "current-release/mouse_ccf/ara_nissl/ara_nissl_10.nrrd"
)


def find_atlas_dir() -> Path:
    """Locate the BrainGlobe allen_mouse_10um atlas directory."""
    candidates = [
        Path.home() / ".brainglobe" / "allen_mouse_10um_v1.2",
        Path.home() / ".brainglobe" / "allen_mouse_10um_v1.1",
        Path.home() / ".brainglobe" / "allen_mouse_10um_v1.0",
    ]
    for p in candidates:
        if p.exists() and (p / "metadata.json").exists():
            return p
    raise FileNotFoundError(
        "Could not find BrainGlobe allen_mouse_10um atlas. "
        "Run: python -c \"from brainglobe_atlasapi import BrainGlobeAtlas; BrainGlobeAtlas('allen_mouse_10um')\" "
        "to download it first, or pass --atlas-dir explicitly."
    )


def download_with_progress(url: str, dest: Path) -> None:
    """Download a file with progress reporting."""
    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  Downloading: {mb:.0f}/{total_mb:.0f} MB ({pct}%)", end="", flush=True)

    urlretrieve(url, str(dest), reporthook=reporthook)
    print()  # newline after progress


def nrrd_to_tiff(nrrd_path: Path, tiff_path: Path, expected_shape: tuple[int, int, int]) -> None:
    """Convert an NRRD volume to TIFF, matching BrainGlobe's format."""
    import SimpleITK as sitk
    import tifffile

    logger.info("Reading NRRD: %s", nrrd_path)
    img = sitk.ReadImage(str(nrrd_path))
    arr = sitk.GetArrayFromImage(img)

    logger.info("NRRD shape: %s, dtype: %s", arr.shape, arr.dtype)

    # The Allen NRRD is in PIR orientation (Posterior-Inferior-Right).
    # BrainGlobe allen_mouse_10um uses ASR (Anterior-Superior-Right).
    # The axes need to be reoriented to match the existing reference.tiff.
    # BrainGlobe stores volumes as (AP, DV, LR) = (z, y, x) in ASR.
    #
    # SimpleITK reads the NRRD with proper orientation metadata,
    # but the raw array may need transposing. Check against expected shape.
    if arr.shape != expected_shape:
        logger.info(
            "Shape mismatch: got %s, expected %s. Attempting reorientation...",
            arr.shape,
            expected_shape,
        )
        # Try common axis permutations
        import numpy as np

        for axes in [(0, 1, 2), (2, 1, 0), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1)]:
            candidate = np.transpose(arr, axes)
            if candidate.shape == expected_shape:
                arr = candidate
                logger.info("Reoriented with axes=%s -> shape %s", axes, arr.shape)
                break
        else:
            # If simple transposition doesn't work, use SimpleITK reorientation
            logger.info("Transposition insufficient, using SimpleITK DICOMOrient...")
            # Resample to ASR orientation
            img_reoriented = sitk.DICOMOrient(img, "ASR")
            arr = sitk.GetArrayFromImage(img_reoriented)
            if arr.shape != expected_shape:
                logger.warning(
                    "After reorientation, shape is %s (expected %s). "
                    "The volume may need manual inspection.",
                    arr.shape,
                    expected_shape,
                )

    # Ensure uint16 (BrainGlobe reference volumes are typically uint16)
    if arr.dtype != "uint16":
        import numpy as np
        if arr.max() > 65535:
            arr = (arr / arr.max() * 65535).astype(np.uint16)
        else:
            arr = arr.astype(np.uint16)

    logger.info("Writing TIFF: %s (shape=%s, dtype=%s)", tiff_path, arr.shape, arr.dtype)
    tifffile.imwrite(str(tiff_path), arr)


def update_metadata(atlas_dir: Path, ref_name: str) -> None:
    """Add reference name to metadata.json additional_references list."""
    meta_path = atlas_dir / "metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)

    refs = metadata.get("additional_references", [])
    if ref_name not in refs:
        refs.append(ref_name)
        metadata["additional_references"] = refs
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Updated metadata.json: additional_references=%s", refs)
    else:
        logger.info("'%s' already in additional_references", ref_name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Allen Nissl volume and install as BrainGlobe additional reference.",
    )
    parser.add_argument(
        "--atlas-dir",
        type=Path,
        default=None,
        help="BrainGlobe atlas directory (default: auto-detect).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if nissl.tiff already exists.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Find atlas directory
    if args.atlas_dir:
        atlas_dir = args.atlas_dir
    else:
        try:
            atlas_dir = find_atlas_dir()
        except FileNotFoundError as e:
            logger.error("%s", e)
            return 1

    logger.info("Atlas directory: %s", atlas_dir)

    # Check if already installed
    tiff_path = atlas_dir / "nissl.tiff"
    if tiff_path.exists() and not args.force:
        logger.info("nissl.tiff already exists (%s). Use --force to re-download.", tiff_path)
        update_metadata(atlas_dir, "nissl")
        logger.info("Done.")
        return 0

    # Read expected shape from metadata
    meta_path = atlas_dir / "metadata.json"
    with open(meta_path) as f:
        metadata = json.load(f)
    expected_shape = tuple(metadata["shape"])
    logger.info("Expected volume shape: %s", expected_shape)

    # Download NRRD to temp file
    with tempfile.TemporaryDirectory() as tmpdir:
        nrrd_path = Path(tmpdir) / "ara_nissl_10.nrrd"
        logger.info("Downloading Nissl volume from Allen Institute...")
        logger.info("  URL: %s", NISSL_URL)
        try:
            download_with_progress(NISSL_URL, nrrd_path)
        except Exception as e:
            logger.error("Download failed: %s", e)
            return 1

        nrrd_size_mb = nrrd_path.stat().st_size / (1024 * 1024)
        logger.info("Downloaded: %.0f MB", nrrd_size_mb)

        # Convert NRRD -> TIFF
        try:
            nrrd_to_tiff(nrrd_path, tiff_path, expected_shape)
        except Exception as e:
            logger.error("Conversion failed: %s", e)
            return 1

    tiff_size_mb = tiff_path.stat().st_size / (1024 * 1024)
    logger.info("Installed: %s (%.0f MB)", tiff_path, tiff_size_mb)

    # Update metadata.json
    update_metadata(atlas_dir, "nissl")

    logger.info("Done. You can now use --reference nissl with generate_training_data.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
