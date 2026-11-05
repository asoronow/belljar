"""Alignment pipeline step.

Combines atlas slicing, registration, and annotation warping into a single
pipeline step that processes a batch of tissue sections.

Outputs are flat files named ``<Kind>_<section>.<ext>`` in the output
directory, the layout the count step and the v1 adjust/count tools pair
by section stem.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from belljar.atlas.slicer import slice_atlas_and_annotation
from belljar.io.formats import list_images, read_image
from belljar.pipeline.base import PipelineStep, ProgressCallback
from belljar.pipeline.params import AlignParams
from belljar.registration.classical import register_to_atlas
from belljar.types import StepResult

logger = logging.getLogger(__name__)


class AlignStep(PipelineStep[AlignParams]):
    """Register tissue sections to atlas slices."""

    params_class = AlignParams

    @property
    def name(self) -> str:
        return "Align"

    def validate_inputs(self, params: AlignParams) -> list[str]:
        errors: list[str] = []
        if not params.input_dir.is_dir():
            errors.append(f"Input directory does not exist: {params.input_dir}")
        if not params.alignments:
            errors.append("alignments list is empty")
        return errors

    def run(self, progress: ProgressCallback, params: AlignParams) -> StepResult:
        input_dir = params.input_dir
        output_dir = params.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        alignments = params.alignments
        atlas_provider = self.atlas_provider

        files = list_images(input_dir)
        if not files:
            return StepResult(
                success=False, errors=["No image files found in input directory"]
            )

        warnings: list[str] = []
        metrics_all: list[dict[str, Any]] = []
        processed = 0

        for i, alignment in enumerate(alignments):
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

                stem = alignment.section_name
                np.save(str(output_dir / f"Annotation_{stem}.npy"), warped_labels)
                cv2.imwrite(str(output_dir / f"Atlas_{stem}.png"), warped_atlas)
                cv2.imwrite(str(output_dir / f"Label_{stem}.png"), color_label)
                composite = cv2.addWeighted(
                    cv2.cvtColor(tissue, cv2.COLOR_GRAY2RGB), 0.8, color_label, 0.2, 0
                )
                cv2.imwrite(str(output_dir / f"Composite_{stem}.png"), composite)

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
