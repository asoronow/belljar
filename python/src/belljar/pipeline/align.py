"""Alignment pipeline step.

Combines atlas slicing, registration, and annotation warping into a single
pipeline step that processes a batch of tissue sections.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from belljar.atlas.provider import AtlasProvider
from belljar.atlas.slicer import slice_atlas_and_annotation
from belljar.config import BelljarConfig
from belljar.io.formats import list_images, read_image
from belljar.pipeline.base import PipelineStep, ProgressCallback
from belljar.registration.classical import register_to_atlas
from belljar.types import SliceAlignment, StepResult

logger = logging.getLogger(__name__)


class AlignStep(PipelineStep):
    """Register tissue sections to atlas slices."""

    @property
    def name(self) -> str:
        return "Align"

    def validate_inputs(self, **kwargs: Any) -> list[str]:
        errors: list[str] = []
        input_dir = kwargs.get("input_dir")
        output_dir = kwargs.get("output_dir")
        alignments = kwargs.get("alignments")

        if not input_dir:
            errors.append("input_dir is required")
        elif not Path(input_dir).is_dir():
            errors.append(f"Input directory does not exist: {input_dir}")

        if not output_dir:
            errors.append("output_dir is required")

        if not alignments:
            errors.append("alignments list is required")

        return errors

    def run(self, progress: ProgressCallback, **kwargs: Any) -> StepResult:
        input_dir = Path(kwargs["input_dir"])
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        alignments: list[dict[str, Any]] = kwargs["alignments"]

        # Load atlas
        atlas_provider = AtlasProvider(
            self.config.atlas.atlas_name,
            reference_name=self.config.atlas.reference_name,
        )

        files = list_images(input_dir)
        if not files:
            return StepResult(
                success=False, errors=["No image files found in input directory"]
            )

        warnings: list[str] = []
        metrics_all: list[dict[str, float]] = []
        processed = 0

        for i, alignment_data in enumerate(alignments):
            alignment = SliceAlignment(**alignment_data)
            progress(i, len(alignments), f"Registering {alignment.section_name}")

            # Find matching file
            matched_file = None
            for f in files:
                if f.stem == alignment.section_name:
                    matched_file = f
                    break

            if matched_file is None:
                warnings.append(f"No image file found for {alignment.section_name}")
                continue

            try:
                tissue = read_image(matched_file, grayscale=True)

                # Slice atlas at estimated position and angles
                atlas_slice, annotation_slice = slice_atlas_and_annotation(
                    atlas_provider.reference,
                    atlas_provider.annotation,
                    int(round(alignment.ap_position)),
                    alignment.x_angle,
                    alignment.y_angle,
                    alignment.z_angle,
                )

                # Register tissue to atlas slice
                warped_labels, warped_atlas, color_label, reg_metrics = register_to_atlas(
                    tissue,
                    atlas_slice,
                    annotation_slice,
                    atlas_provider.structure_map,
                    self.config.registration,
                )

                # Save outputs
                section_dir = output_dir / alignment.section_name
                section_dir.mkdir(exist_ok=True)

                np.save(str(section_dir / "annotation.npy"), warped_labels)
                cv2.imwrite(str(section_dir / "atlas.png"), warped_atlas)
                cv2.imwrite(str(section_dir / "color_label.png"), color_label)

                # Save annotation as pickle for backward compatibility
                with open(section_dir / "annotation.pkl", "wb") as f:
                    pickle.dump(warped_labels, f)

                metrics_all.append({
                    "section": alignment.section_name,
                    "mutual_information": reg_metrics.mutual_information,
                    "ncc": reg_metrics.normalized_cross_correlation,
                })
                processed += 1

            except Exception as e:
                warnings.append(f"Failed to register {alignment.section_name}: {e}")
                logger.exception("Registration failed for %s", alignment.section_name)

        progress(len(alignments), len(alignments), "Done")

        return StepResult(
            success=processed > 0,
            output_path=str(output_dir),
            metrics={
                "sections_processed": processed,
                "sections_total": len(alignments),
                "per_section": metrics_all,
            },
            warnings=warnings,
        )
