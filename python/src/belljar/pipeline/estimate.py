"""Estimation pipeline step.

Runs the slice position estimator on a batch of tissue section images,
producing alignment parameters (AP position, angles, anchoring vectors)
for downstream registration.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from belljar.config import BelljarConfig
from belljar.estimation.model_manager import ensure_model
from belljar.estimation.postprocess import (
    denormalize_predictions,
    enforce_orthogonality,
    integrate_angles,
    regularize_spacing,
)
from belljar.estimation.predictor import (
    load_model,
    predict_slice_position,
    predict_with_uncertainty,
)
from belljar.io.formats import list_images, read_image
from belljar.pipeline.base import PipelineStep, ProgressCallback
from belljar.types import StepResult

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    """JSON encoder for numpy types."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class EstimateStep(PipelineStep):
    """Estimate slice positions and orientations for tissue sections."""

    @property
    def name(self) -> str:
        return "Estimate"

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

        # Model must be resolvable (explicit path, cached, or downloadable)
        model_path = kwargs.get("model_path")
        if model_path and not Path(model_path).exists():
            errors.append(f"Model file not found: {model_path}")

        return errors

    def run(self, progress: ProgressCallback, **kwargs: Any) -> StepResult:
        input_dir = Path(kwargs["input_dir"])
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path_arg = kwargs.get("model_path")
        gcs_model_uri = kwargs.get("gcs_model_uri")
        use_uncertainty = kwargs.get("uncertainty", False)

        # Resolve model
        try:
            model_path = ensure_model(
                model_path=Path(model_path_arg) if model_path_arg else None,
                gcs_uri=gcs_model_uri,
            )
        except FileNotFoundError as e:
            return StepResult(success=False, errors=[str(e)])

        # Load model
        estimation_config = self.config.estimation
        model = load_model(model_path, estimation_config)
        logger.info("Model loaded from %s", model_path)

        # List input images
        files = list_images(input_dir)
        if not files:
            return StepResult(
                success=False, errors=["No image files found in input directory"]
            )

        progress(0, len(files), "Starting estimation")

        # Run inference on each section
        predictions: list[dict[str, Any]] = []
        warnings: list[str] = []

        for i, img_path in enumerate(files):
            progress(i, len(files), f"Estimating {img_path.stem}")
            try:
                image = read_image(img_path, grayscale=True)

                if use_uncertainty:
                    result = predict_with_uncertainty(
                        image, model, estimation_config
                    )
                    pred = result["prediction"]
                    pred["uncertainty"] = result["uncertainty"].tolist()
                else:
                    pred = predict_slice_position(image, model, estimation_config)

                pred["section_name"] = img_path.stem
                predictions.append(pred)

            except Exception as e:
                warnings.append(f"Failed to estimate {img_path.stem}: {e}")
                logger.exception("Estimation failed for %s", img_path.stem)

        if not predictions:
            return StepResult(
                success=False,
                errors=["All estimations failed"],
                warnings=warnings,
            )

        # Postprocess: denormalize → smooth angles → regularize spacing → orthogonalize
        predictions = denormalize_predictions(predictions)
        predictions = integrate_angles(predictions)
        predictions = regularize_spacing(predictions, method="ransac")
        predictions = enforce_orthogonality(predictions)

        # Build alignments list for downstream steps
        alignments = []
        for pred in predictions:
            alignments.append({
                "section_name": pred["section_name"],
                "ap_position": pred["z_position"],
                "x_angle": pred["x_angle"],
                "y_angle": pred["y_angle"],
                "z_angle": 0.0,
                "region": "A",
                "hemisphere": "W",
                "linked": True,
                "anchoring_vectors": pred.get("anchoring_vectors"),
            })

        # Save output
        output_file = output_dir / "estimates.json"
        output_data = {
            "alignments": alignments,
            "metadata": {
                "total_sections": len(alignments),
                "model_path": str(model_path),
                "preprocessing": estimation_config.preprocessing,
                "uncertainty": use_uncertainty,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        }
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2, default=_json_default)

        progress(len(files), len(files), "Done")

        return StepResult(
            success=True,
            output_path=str(output_file),
            metrics={
                "sections_estimated": len(predictions),
                "sections_total": len(files),
                "sections_failed": len(files) - len(predictions),
            },
            warnings=warnings,
        )
