"""Cell detection pipeline step.

Runs YOLO + SAHI tiled detection on tissue images, with post-processing
(area filtering, statistical outlier removal, eccentricity filtering).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import tifffile as tiff

from belljar.config import BelljarConfig
from belljar.detection.detector import (
    create_detection_model,
    detect_neurons,
    export_bboxes,
    load_image,
    _prepare_for_detection,
)
from belljar.io.formats import list_images
from belljar.pipeline.base import PipelineStep, ProgressCallback
from belljar.types import DetectionResult, StepResult

logger = logging.getLogger(__name__)


class DetectStep(PipelineStep):
    """Detect cells/neurons in tissue images using YOLO + SAHI."""

    @property
    def name(self) -> str:
        return "Detect"

    def validate_inputs(self, **kwargs: Any) -> list[str]:
        errors: list[str] = []
        input_dir = kwargs.get("input_dir")
        output_dir = kwargs.get("output_dir")
        model_path = kwargs.get("model_path")

        if not input_dir:
            errors.append("input_dir is required")
        elif not Path(input_dir).is_dir():
            errors.append(f"Input directory does not exist: {input_dir}")

        if not output_dir:
            errors.append("output_dir is required")

        if not model_path:
            errors.append("model_path is required")
        elif not Path(model_path).is_file():
            errors.append(f"Model file does not exist: {model_path}")

        return errors

    def run(self, progress: ProgressCallback, **kwargs: Any) -> StepResult:
        input_dir = Path(kwargs["input_dir"])
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = Path(kwargs["model_path"])
        save_visualizations: bool = kwargs.get("save_visualizations", True)

        config = self.config.detection

        # Load model once
        model = create_detection_model(model_path, config)

        files = list_images(input_dir)
        if not files:
            return StepResult(
                success=False,
                errors=["No supported image files found in input directory"],
            )

        warnings: list[str] = []
        total_detections = 0
        processed = 0

        for i, file_path in enumerate(files):
            progress(i, len(files), f"Detecting in {file_path.name}")
            try:
                img, index_order = load_image(file_path)

                # Handle multi-channel images
                if img.ndim == 3 and index_order == "C":
                    # Channel-first TIFF (z/c, y, x)
                    channels = [img[c] for c in range(img.shape[0])]
                elif img.ndim == 3 and index_order == "F":
                    # Channel-last (y, x, c) — typically BGR from cv2
                    channels = [img[:, :, c] for c in range(img.shape[2])]
                else:
                    channels = [img]

                channel_results: list[DetectionResult] = []
                for ch_idx, channel_img in enumerate(channels):
                    result = detect_neurons(channel_img, model, config, channel_index=ch_idx)
                    channel_results.append(result)
                    total_detections += result.count

                # Save results
                results_path = output_dir / f"{file_path.stem}.pkl"
                with open(results_path, "wb") as f:
                    pickle.dump(channel_results, f)

                # Optionally save visualization
                if save_visualizations and channel_results:
                    for ch_idx, result in enumerate(channel_results):
                        if result.boxes:
                            vis_img = _prepare_for_detection(channels[ch_idx])
                            vis_path = output_dir / f"{file_path.stem}_ch{ch_idx}_detections.png"
                            export_bboxes(vis_img, result.boxes, vis_path)

                processed += 1

            except Exception as e:
                warnings.append(f"Failed to process {file_path.name}: {e}")
                logger.exception("Detection failed for %s", file_path.name)

        progress(len(files), len(files), "Done")

        return StepResult(
            success=processed > 0,
            output_path=str(output_dir),
            metrics={
                "files_processed": processed,
                "files_total": len(files),
                "total_detections": total_detections,
            },
            warnings=warnings,
        )
