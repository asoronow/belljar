"""Estimation pipeline step.

Runs the slice position estimator on a batch of tissue section images,
producing alignment parameters (AP position, angles, anchoring vectors)
for downstream registration.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

from belljar.estimation.model_manager import ensure_model
from belljar.estimation.postprocess import (
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
from belljar.pipeline.params import EstimateParams
from belljar.types import StepResult

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    """JSON encoder for numpy types."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _direction_uncertainty(per_output_std: list[float]) -> float:
    """Collapse per-output MC std into one scalar for angle smoothing.

    Angles derive from the u and v direction vectors (outputs 3..8), so the
    origin components are excluded. A 3-output legacy model has no direction
    vectors; its angle outputs (1..2) are used instead.
    """
    std = np.asarray(per_output_std, dtype=np.float64)
    if std.size == 9:
        return float(std[3:].mean())
    if std.size == 3:
        return float(std[1:].mean())
    raise ValueError(f"Expected 9 or 3 model outputs, got {std.size}")


class EstimateStep(PipelineStep[EstimateParams]):
    """Estimate slice positions and orientations for tissue sections."""

    params_class = EstimateParams

    @property
    def name(self) -> str:
        return "Estimate"

    def validate_inputs(self, params: EstimateParams) -> list[str]:
        errors: list[str] = []
        if not params.input_dir.is_dir():
            errors.append(f"Input directory does not exist: {params.input_dir}")
        # A model given explicitly must exist; otherwise the cache or GCS is tried at run time.
        if params.model_path is not None and not params.model_path.exists():
            errors.append(f"Model file not found: {params.model_path}")
        return errors

    def run(self, progress: ProgressCallback, params: EstimateParams) -> StepResult:
        input_dir = params.input_dir
        output_dir = params.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        use_uncertainty = params.uncertainty

        try:
            model_path = ensure_model(
                model_path=params.model_path,
                gcs_uri=params.gcs_model_uri,
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
                    pred = predict_with_uncertainty(image, model, estimation_config)
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

        # Predictions are already in physical units; smooth angles, regularize
        # spacing, then orthogonalize.
        angle_uncertainties = None
        if use_uncertainty:
            angle_uncertainties = [
                _direction_uncertainty(pred["uncertainty"]) for pred in predictions
            ]
        predictions = integrate_angles(predictions, uncertainties=angle_uncertainties)
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
            if "uncertainty" in pred:
                alignments[-1]["uncertainty"] = pred["uncertainty"]

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
