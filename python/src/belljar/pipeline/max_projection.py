"""Max projection pipeline step.

Collapses z-stack images along the smallest dimension (channel/z) to produce
a single 2D image per file.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import tifffile as tiff

from belljar.io.formats import list_images
from belljar.pipeline.base import PipelineStep, ProgressCallback
from belljar.pipeline.params import MaxProjectionParams
from belljar.types import StepResult

logger = logging.getLogger(__name__)


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

        files = list_images(input_dir)
        if not files:
            return StepResult(
                success=False,
                errors=["No supported image files found in input directory"],
            )

        warnings: list[str] = []
        processed = 0

        for i, file_path in enumerate(files):
            progress(i, len(files), f"Processing {file_path.name}")
            try:
                img = tiff.imread(str(file_path))
                if img.ndim < 3:
                    # Already 2D, just copy
                    cv2.imwrite(str(output_dir / f"{file_path.stem}.tif"), img)
                else:
                    channel_dim = int(np.argmin(img.shape))
                    projected = np.max(img, axis=channel_dim)
                    cv2.imwrite(str(output_dir / f"{file_path.stem}.tif"), projected)
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
