"""Results collation pipeline step.

Merges per-section count CSV files or count results across an experiment
into a single summary output. Replaces the old Tkinter-based collate.py.
"""

from __future__ import annotations

import csv
import logging
from typing import TYPE_CHECKING, Any

from belljar.pipeline.base import PipelineStep, ProgressCallback
from belljar.pipeline.params import CollateParams
from belljar.types import StepResult

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


def collate_counts(
    input_csv: Path,
    structure_map: dict[int, dict[str, Any]],
    output_path: Path,
    include_layers: bool = False,
) -> dict[str, int]:
    """Collate counts from a per-section CSV into region totals.

    Args:
        input_csv: Path to the count_results.csv from the Count step.
        structure_map: Region ID -> metadata dict.
        output_path: Path for the collated output CSV.
        include_layers: Whether to include layer-level regions.

    Returns:
        Dict of region acronym -> total count.
    """
    # Read the count results
    section_data: dict[str, dict[str, list[int]]] = {}
    current_section: str | None = None
    header_seen = False

    with open(input_csv) as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                current_section = None
                header_seen = False
                continue

            # Stop at totals/colocalization sections
            if row[0] in ("Totals", "Colocalization Matrix (by Section)"):
                break

            if not header_seen and row[0] not in ("Region Acronym",):
                # This is a section name row
                current_section = row[0]
                section_data[current_section] = {}
                continue

            if row[0] == "Region Acronym":
                header_seen = True
                continue

            if current_section and header_seen:
                acronym = row[0]
                counts = [int(x) for x in row[3:] if x]
                section_data[current_section][acronym] = counts

    # Aggregate
    totals: dict[str, int] = {}
    for section_counts in section_data.values():
        for acronym, counts in section_counts.items():
            if not include_layers:
                # Check if this is a layer region and skip
                region_id = None
                for rid, info in structure_map.items():
                    if info["acronym"] == acronym:
                        region_id = rid
                        break
                if region_id is not None and "layer" in structure_map[region_id]["name"].lower():
                    # Roll up to parent
                    id_path = structure_map[region_id].get("id_path", "").split("/")
                    if len(id_path) >= 2:
                        parent_id = int(id_path[-2])
                        if parent_id in structure_map:
                            acronym = structure_map[parent_id]["acronym"]

            total = sum(counts)
            totals[acronym] = totals.get(acronym, 0) + total

    # Write collated output
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Region Acronym", "Region Name", "Total Count"])

        # Build reverse lookup
        acronym_to_id: dict[str, int] = {}
        for k, v in structure_map.items():
            acronym_to_id[v["acronym"]] = k

        for acronym in sorted(totals):
            region_id = acronym_to_id.get(acronym)
            region_name = (
                structure_map[region_id]["name"] if region_id in structure_map else "Unknown"
            )
            writer.writerow([acronym, region_name, totals[acronym]])

    return totals


class CollateStep(PipelineStep[CollateParams]):
    """Collate counting results across sections into a summary."""

    params_class = CollateParams

    @property
    def name(self) -> str:
        return "Collate"

    def validate_inputs(self, params: CollateParams) -> list[str]:
        if not params.input_csv.is_file():
            return [f"Input CSV does not exist: {params.input_csv}"]
        return []

    def run(self, progress: ProgressCallback, params: CollateParams) -> StepResult:
        input_csv = params.input_csv
        output_dir = params.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        structure_map = self.atlas_provider.structure_map
        include_layers = params.include_layers

        progress(0, 2, "Collating results")

        try:
            output_path = output_dir / "collated_results.csv"
            totals = collate_counts(
                input_csv, structure_map, output_path, include_layers
            )

            total_count = sum(totals.values())
            num_regions = len(totals)

            progress(2, 2, "Done")

            return StepResult(
                success=True,
                output_path=str(output_path),
                metrics={
                    "total_count": total_count,
                    "regions_with_counts": num_regions,
                },
            )
        except Exception as e:
            logger.exception("Collation failed")
            return StepResult(success=False, errors=[str(e)])
