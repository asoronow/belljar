"""Image sharpening pipeline step.

Applies CLAHE contrast enhancement, unsharp masking, and white tophat
filtering to improve tissue visibility before registration/detection.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile as tiff
from numpy.typing import NDArray
from skimage.filters import unsharp_mask
from skimage.morphology import disk, white_tophat

from belljar.config import BelljarConfig
from belljar.io.formats import list_images
from belljar.pipeline.base import PipelineStep, ProgressCallback
from belljar.types import StepResult

logger = logging.getLogger(__name__)


def enhance_contrast(image: NDArray, saturation_level: float = 0.05) -> NDArray:
    """Enhance contrast by saturating a percentage of pixels at both ends.

    Args:
        image: Input image (integer dtype).
        saturation_level: Fraction of pixels to saturate (0-100 scale).

    Returns:
        Contrast-enhanced image with same dtype.
    """
    saturation_point = saturation_level / 100.0
    flat = image.ravel()

    low = np.percentile(flat, saturation_point)
    high = np.percentile(flat, 100.0 - saturation_point)
    clipped = np.clip(flat, low, high)

    if np.issubdtype(image.dtype, np.integer):
        dtype_min = np.iinfo(image.dtype).min
        dtype_max = np.iinfo(image.dtype).max
    else:
        dtype_min = float(np.finfo(image.dtype).min)
        dtype_max = float(np.finfo(image.dtype).max)

    rescaled = np.interp(clipped, (clipped.min(), clipped.max()), (dtype_min, dtype_max))
    return rescaled.reshape(image.shape).astype(image.dtype)


def sharpen_image(
    img: NDArray,
    *,
    equalize: bool = False,
    radius: float = 3.0,
    amount: float = 2.0,
    tophat_radius: int = 15,
) -> NDArray:
    """Apply sharpening pipeline to a single image.

    Args:
        img: Input image (uint8 or uint16).
        equalize: Whether to apply CLAHE + contrast enhancement first.
        radius: Unsharp mask radius.
        amount: Unsharp mask amount.
        tophat_radius: Disk radius for white tophat.

    Returns:
        Sharpened image with original dtype.
    """
    original_dtype = img.dtype

    if equalize:
        if img.dtype == np.uint16:
            img_8bit = (img / 256).astype(np.uint8)
        else:
            img_8bit = img.astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        img = clahe.apply(img_8bit)
        img = enhance_contrast(img)

    img = unsharp_mask(img, radius=radius, amount=amount, preserve_range=True)
    img = white_tophat(img, disk(tophat_radius))

    return img.astype(original_dtype)


class SharpenStep(PipelineStep):
    """Sharpen tissue images for better registration and detection."""

    @property
    def name(self) -> str:
        return "Sharpen"

    def validate_inputs(self, **kwargs: Any) -> list[str]:
        errors: list[str] = []
        input_dir = kwargs.get("input_dir")
        output_dir = kwargs.get("output_dir")

        if not input_dir:
            errors.append("input_dir is required")
        elif not Path(input_dir).is_dir():
            errors.append(f"Input directory does not exist: {input_dir}")

        if not output_dir:
            errors.append("output_dir is required")

        return errors

    def run(self, progress: ProgressCallback, **kwargs: Any) -> StepResult:
        input_dir = Path(kwargs["input_dir"])
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        equalize: bool = kwargs.get("equalize", False)
        radius: float = kwargs.get("radius", 3.0)
        amount: float = kwargs.get("amount", 2.0)

        files = list_images(input_dir)
        if not files:
            return StepResult(
                success=False,
                errors=["No supported image files found in input directory"],
            )

        warnings: list[str] = []
        processed = 0

        for i, file_path in enumerate(files):
            progress(i, len(files), f"Sharpening {file_path.name}")
            try:
                img = tiff.imread(str(file_path))
                result = sharpen_image(img, equalize=equalize, radius=radius, amount=amount)
                out_name = f"{file_path.stem}{file_path.suffix}"
                cv2.imwrite(str(output_dir / out_name), result)
                processed += 1
            except Exception as e:
                warnings.append(f"Failed to process {file_path.name}: {e}")
                logger.warning("Failed to process %s: %s", file_path.name, e)

        progress(len(files), len(files), "Done")

        return StepResult(
            success=processed > 0,
            output_path=str(output_dir),
            metrics={"files_processed": processed, "files_total": len(files)},
            warnings=warnings,
        )
