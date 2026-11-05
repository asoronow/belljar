"""Cell detection pipeline step.

Runs YOLO + SAHI tiled detection on tissue images, with post-processing
(area filtering, statistical outlier removal, eccentricity filtering).
"""

from __future__ import annotations

import logging
import pickle

from belljar.detection.detector import (
    _prepare_for_detection,
    create_detection_model,
    detect_neurons,
    export_bboxes,
    load_image,
)
from belljar.io.formats import list_images
from belljar.pipeline.base import PipelineStep, ProgressCallback
from belljar.pipeline.params import DetectParams
from belljar.types import DetectionResult, StepResult

logger = logging.getLogger(__name__)


class DetectStep(PipelineStep[DetectParams]):
    """Detect cells/neurons in tissue images using YOLO + SAHI."""

    params_class = DetectParams

    @property
    def name(self) -> str:
        return "Detect"

    def validate_inputs(self, params: DetectParams) -> list[str]:
        errors: list[str] = []
        if not params.input_dir.is_dir():
            errors.append(f"Input directory does not exist: {params.input_dir}")
        if not params.model_path.is_file():
            errors.append(f"Model file does not exist: {params.model_path}")
        return errors

    def run(self, progress: ProgressCallback, params: DetectParams) -> StepResult:
        input_dir = params.input_dir
        output_dir = params.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = params.model_path
        save_visualizations = params.save_visualizations

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
