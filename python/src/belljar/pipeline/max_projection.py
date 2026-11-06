"""Max projection pipeline step.

Collapses z-stack images to a single YX plane per file, using the TIFF's
axis metadata to decide which dimensions are stack dimensions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import tifffile as tiff

from belljar.io.formats import check_unique_stems, list_tiffs, section_stem
from belljar.pipeline.base import PipelineStep, ProgressCallback
from belljar.pipeline.params import MaxProjectionParams
from belljar.types import StepResult

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Axis letters tifffile assigns from explicit metadata, and the letters it
# uses when the file does not say what its leading dimensions are (Q for
# shaped arrays without axes, I for a bare page sequence).
KNOWN_AXES = frozenset("ZCSYX")
UNLABELED_AXES = frozenset("QI")


def max_project(img: NDArray, axes: str, name: str) -> NDArray:
    """Collapse an image to a single YX plane.

    With axis metadata, every Z, C and S axis is max-projected away. Without
    it, a 2-D image is returned as is and a 3-D image is projected over its
    first axis, which must be the smallest.

    Raises:
        ValueError: If the layout cannot be determined from metadata or shape.
    """
    if set(axes) <= KNOWN_AXES and "Y" in axes and "X" in axes:
        stack_axes = tuple(i for i, axis in enumerate(axes) if axis not in "YX")
        return np.max(img, axis=stack_axes) if stack_axes else img
    if not set(axes) <= KNOWN_AXES | UNLABELED_AXES:
        raise ValueError(
            f"{name} has unsupported axes '{axes}'; only Z, C and S stack axes can be projected"
        )
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[0] == min(img.shape):
        return np.max(img, axis=0)
    raise ValueError(
        f"{name} has shape {img.shape} and no axis metadata; "
        "a stack must be ZYX with Z as the smallest dimension"
    )


def max_project_file(path: Path) -> NDArray:
    """Read a TIFF and collapse it to a single YX plane, keeping its dtype."""
    with tiff.TiffFile(str(path)) as handle:
        series = handle.series[0]
        return max_project(series.asarray(), series.axes, path.name)


class MaxProjectionStep(PipelineStep[MaxProjectionParams]):
    """Collapse z-stack images via max-intensity projection."""

    params_class = MaxProjectionParams

    @property
    def name(self) -> str:
        return "Max Projection"

    def validate_inputs(self, params: MaxProjectionParams) -> list[str]:
        if not params.input_dir.is_dir():
            return [f"Input directory does not exist: {params.input_dir}"]
        return []

    def run(self, progress: ProgressCallback, params: MaxProjectionParams) -> StepResult:
        input_dir = params.input_dir
        output_dir = params.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        files = list_tiffs(input_dir)
        if not files:
            return StepResult(
                success=False,
                errors=["No TIFF files found in input directory"],
            )
        try:
            check_unique_stems(files)
        except ValueError as e:
            return StepResult(success=False, errors=[str(e)])

        warnings: list[str] = []
        processed = 0

        for i, file_path in enumerate(files):
            progress(i, len(files), f"Processing {file_path.name}")
            try:
                projected = max_project_file(file_path)
                tiff.imwrite(output_dir / f"{section_stem(file_path)}.tif", projected)
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
