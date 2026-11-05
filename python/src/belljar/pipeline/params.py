"""Typed parameters for each pipeline step.

The CLI and the JSON-RPC server both build these models, so a request is
checked once, before a step runs, and every step reads validated fields
instead of raw keyword arguments.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from belljar.types import SliceAlignment


class StepParams(BaseModel):
    """Base for step parameter models; unknown fields are an error."""

    model_config = ConfigDict(extra="forbid")


class MaxProjectionParams(StepParams):
    input_dir: Path
    output_dir: Path


class SharpenParams(StepParams):
    input_dir: Path
    output_dir: Path
    equalize: bool = False
    radius: float = 3.0
    amount: float = 2.0


class EstimateParams(StepParams):
    input_dir: Path
    output_dir: Path
    model_path: Path | None = None
    gcs_model_uri: str | None = None
    uncertainty: bool = False


class AlignParams(StepParams):
    input_dir: Path
    output_dir: Path
    alignments: list[SliceAlignment]


class DetectParams(StepParams):
    input_dir: Path
    output_dir: Path
    model_path: Path
    save_visualizations: bool = True


class CountParams(StepParams):
    predictions_dir: Path
    annotations_dir: Path
    output_dir: Path
    include_layers: bool = False


class CollateParams(StepParams):
    input_csv: Path
    output_dir: Path
    include_layers: bool = False
