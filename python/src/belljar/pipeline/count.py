"""Cell counting pipeline step.

Integrates detection results with warped annotation maps to count cells
per brain region, with optional layer-level counting and colocalization
analysis for multi-channel data.
"""

from __future__ import annotations

import csv
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from belljar.config import BelljarConfig
from belljar.pipeline.base import PipelineStep, ProgressCallback
from belljar.registration.preprocessing import resize_nearest_neighbor
from belljar.types import DetectionResult, StepResult

logger = logging.getLogger(__name__)


def iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute Intersection over Union between two bounding boxes.

    Args:
        box_a: [x1, y1, x2, y2] for the first box.
        box_b: [x1, y1, x2, y2] for the second box.

    Returns:
        IoU value in [0, 1].
    """
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)
    area_a = (box_a[2] - box_a[0] + 1) * (box_a[3] - box_a[1] + 1)
    area_b = (box_b[2] - box_b[0] + 1) * (box_b[3] - box_b[1] + 1)

    return inter_area / float(area_a + area_b - inter_area)


def compute_overlaps(boxes1: list[list[float]], boxes2: list[list[float]]) -> NDArray:
    """Compute IoU overlap matrix between two sets of bounding boxes.

    Returns:
        NxM overlap matrix where [i,j] is the IoU between boxes1[i] and boxes2[j].
    """
    overlaps = np.zeros((len(boxes1), len(boxes2)))
    for i, b1 in enumerate(boxes1):
        for j, b2 in enumerate(boxes2):
            overlaps[i, j] = iou(b1, b2)
    return overlaps


def percent_colocalized(
    boxes1: list[list[float]], boxes2: list[list[float]], threshold: float = 0.5
) -> float:
    """Compute percentage of boxes1 that overlap with any box in boxes2.

    Args:
        boxes1: First set of bounding boxes.
        boxes2: Second set of bounding boxes.
        threshold: Minimum IoU to count as colocalized.

    Returns:
        Percentage of colocalized boxes (0-100).
    """
    if not boxes1 or not boxes2:
        return 0.0
    overlaps = compute_overlaps(boxes1, boxes2)
    max_overlaps = np.max(overlaps, axis=1)
    colocalized_count = int(np.sum(max_overlaps > threshold))
    return (colocalized_count / len(boxes1)) * 100.0


def _get_parent_acronym(
    region_info: dict, regions: dict, include_layers: bool
) -> str:
    """Get the appropriate acronym for counting (region or parent if layer)."""
    if include_layers:
        return region_info["acronym"]
    if "layer" in region_info["name"].lower():
        id_path = region_info.get("id_path", "").split("/")
        if len(id_path) >= 2:
            parent_id = np.uint32(int(id_path[-2]))
            if parent_id in regions:
                return regions[parent_id]["acronym"]
    return region_info["acronym"]


class CountStep(PipelineStep):
    """Count detected cells per brain region."""

    @property
    def name(self) -> str:
        return "Count"

    def validate_inputs(self, **kwargs: Any) -> list[str]:
        errors: list[str] = []
        predictions_dir = kwargs.get("predictions_dir")
        annotations_dir = kwargs.get("annotations_dir")
        output_dir = kwargs.get("output_dir")
        structure_map = kwargs.get("structure_map")

        if not predictions_dir:
            errors.append("predictions_dir is required")
        elif not Path(predictions_dir).is_dir():
            errors.append(f"Predictions directory does not exist: {predictions_dir}")

        if not annotations_dir:
            errors.append("annotations_dir is required")
        elif not Path(annotations_dir).is_dir():
            errors.append(f"Annotations directory does not exist: {annotations_dir}")

        if not output_dir:
            errors.append("output_dir is required")

        if not structure_map:
            errors.append("structure_map is required")

        return errors

    def run(self, progress: ProgressCallback, **kwargs: Any) -> StepResult:
        predictions_dir = Path(kwargs["predictions_dir"])
        annotations_dir = Path(kwargs["annotations_dir"])
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        regions: dict = kwargs["structure_map"]
        include_layers: bool = kwargs.get("include_layers", False)

        # Build reverse lookup
        acronym_to_id: dict[str, Any] = {}
        for k, v in regions.items():
            acronym_to_id[v["acronym"]] = k

        # Find matching files
        pred_files = sorted(
            [p for p in predictions_dir.iterdir() if p.suffix == ".pkl"]
        )
        ann_files = sorted(
            [p for p in annotations_dir.iterdir() if p.suffix in (".pkl", ".npy")]
        )

        if not pred_files:
            return StepResult(success=False, errors=["No prediction files found"])
        if not ann_files:
            return StepResult(success=False, errors=["No annotation files found"])

        warnings: list[str] = []
        sums: dict[str, dict[int, dict[str, int]]] = {}
        colocalized: dict[str, dict[int, dict[int, float]]] = {}
        region_areas: dict[str, dict[str, int]] = {}

        for i, pred_path in enumerate(pred_files):
            section_name = pred_path.stem
            progress(i, len(pred_files), f"Counting {section_name}")

            # Find matching annotation
            ann_path = None
            for af in ann_files:
                if af.stem.startswith(section_name) or af.stem == section_name:
                    ann_path = af
                    break
            if ann_path is None and i < len(ann_files):
                ann_path = ann_files[i]

            if ann_path is None:
                warnings.append(f"No matching annotation for {section_name}")
                continue

            try:
                with open(pred_path, "rb") as f:
                    predictions: list[DetectionResult] = pickle.load(f)

                if ann_path.suffix == ".npy":
                    annotation = np.load(str(ann_path))
                else:
                    with open(ann_path, "rb") as f:
                        annotation = pickle.load(f)

                predicted_size = (predictions[0].image_width, predictions[0].image_height)
                annotation_rescaled = resize_nearest_neighbor(annotation, predicted_size)

                # Compute area per region
                region_areas[section_name] = {}
                unique_ids, counts = np.unique(annotation_rescaled, return_counts=True)
                for uid, count in zip(unique_ids, counts):
                    if uid not in regions:
                        continue
                    acronym = _get_parent_acronym(regions[uid], regions, include_layers)
                    region_areas[section_name][acronym] = (
                        region_areas[section_name].get(acronym, 0) + int(count)
                    )

                # Count detections per region per channel
                sums[section_name] = {}
                all_boxes: dict[int, list[list[float]]] = {}

                for ch_idx, detection in enumerate(predictions):
                    sums[section_name][ch_idx] = {}
                    all_boxes[ch_idx] = detection.boxes

                    for box in detection.boxes:
                        x1, y1, x2, y2 = box
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        # Clamp to annotation bounds
                        cy = min(cy, annotation_rescaled.shape[0] - 1)
                        cx = min(cx, annotation_rescaled.shape[1] - 1)

                        atlas_id = annotation_rescaled[cy, cx]
                        if atlas_id not in regions:
                            continue

                        acronym = _get_parent_acronym(
                            regions[atlas_id], regions, include_layers
                        )
                        sums[section_name][ch_idx][acronym] = (
                            sums[section_name][ch_idx].get(acronym, 0) + 1
                        )

                # Colocalization analysis
                colocalized[section_name] = {}
                for c1, boxes1 in all_boxes.items():
                    colocalized[section_name][c1] = {}
                    for c2, boxes2 in all_boxes.items():
                        colocalized[section_name][c1][c2] = percent_colocalized(boxes1, boxes2)

            except Exception as e:
                warnings.append(f"Failed to count {section_name}: {e}")
                logger.exception("Counting failed for %s", section_name)

        # Write CSV output
        progress(len(pred_files), len(pred_files) + 1, "Writing output")
        csv_path = output_dir / "count_results.csv"
        _write_count_csv(csv_path, sums, colocalized, region_areas, regions, acronym_to_id)

        progress(len(pred_files) + 1, len(pred_files) + 1, "Done")

        # Compute total counts
        total_counts = 0
        for section_channels in sums.values():
            for channel_counts in section_channels.values():
                total_counts += sum(channel_counts.values())

        return StepResult(
            success=True,
            output_path=str(csv_path),
            metrics={
                "sections_counted": len(sums),
                "total_detections_assigned": total_counts,
            },
            warnings=warnings,
        )


def _write_count_csv(
    path: Path,
    sums: dict[str, dict[int, dict[str, int]]],
    colocalized: dict[str, dict[int, dict[int, float]]],
    region_areas: dict[str, dict[str, int]],
    regions: dict,
    acronym_to_id: dict[str, Any],
) -> None:
    """Write count results to a CSV file."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        running_counts: dict[str, int] = {}

        for section_name, channels in sums.items():
            writer.writerow([section_name])
            num_channels = len(channels)
            header = ["Region Acronym", "Region Name", "Area (px)"]
            header += [f"Channel #{c}" for c in range(num_channels)]
            writer.writerow(header)

            all_acronyms: set[str] = set()
            for ch_counts in channels.values():
                all_acronyms.update(ch_counts.keys())

            for acronym in sorted(all_acronyms):
                region_id = acronym_to_id.get(acronym)
                region_name = regions[region_id]["name"] if region_id and region_id in regions else "Unknown"
                area = region_areas.get(section_name, {}).get(acronym, 0)

                row = [acronym, region_name, area]
                for ch in range(num_channels):
                    count = channels[ch].get(acronym, 0)
                    row.append(count)
                    running_counts[acronym] = running_counts.get(acronym, 0) + count
                writer.writerow(row)

            writer.writerow([])

        # Totals
        writer.writerow(["Totals"])
        writer.writerow(["Region Acronym", "Region Name", "Count"])
        for acronym in sorted(running_counts):
            region_id = acronym_to_id.get(acronym)
            region_name = regions[region_id]["name"] if region_id and region_id in regions else "Unknown"
            writer.writerow([acronym, region_name, running_counts[acronym]])

        writer.writerow([])

        # Colocalization
        writer.writerow(["Colocalization Matrix (by Section)"])
        for section_name, section_coloc in colocalized.items():
            num_ch = len(section_coloc)
            writer.writerow([section_name] + [f"Channel #{c}" for c in range(num_ch)])
            for c1 in range(num_ch):
                row = [f"Channel #{c1}"]
                for c2 in range(num_ch):
                    row.append(section_coloc.get(c1, {}).get(c2, 0))
                writer.writerow(row)
