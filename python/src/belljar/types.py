"""Shared data types for Belljar.

These replace the opaque pickle-based serialization with typed, inspectable dataclasses.
Large arrays (annotation maps, images) are still stored as numpy files alongside the
JSON project file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SliceAlignment:
    """Alignment state for a single histological section."""

    section_name: str
    ap_position: float
    x_angle: float
    y_angle: float
    z_angle: float = 0.0
    region: str = "A"  # "A"=All, "C"=Cerebrum, "NC"=NonCerebrum
    hemisphere: str = "W"  # "W"=Whole, "L"=Left
    linked: bool = True
    mask_path: str | None = None


@dataclass
class DetectionResult:
    """Detection results for a single image channel."""

    boxes: list[list[float]]  # [[x1, y1, x2, y2], ...]
    scores: list[float]
    image_width: int
    image_height: int
    channel_index: int = 0
    model_name: str = ""
    confidence_threshold: float = 0.0

    @property
    def count(self) -> int:
        return len(self.boxes)


@dataclass
class RegistrationMetrics:
    """Quality metrics for a registration result."""

    mutual_information: float = 0.0
    normalized_cross_correlation: float = 0.0


@dataclass
class StepResult:
    """Result returned by every pipeline step."""

    success: bool
    output_path: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class BelljarProject:
    """JSON-serializable project state.

    Large arrays (annotation maps) are stored as .npy files in the output directory
    and referenced by path — not embedded in the project file.
    """

    version: str = "2.0"
    name: str = ""
    input_path: str = ""
    output_path: str = ""
    atlas_name: str = "allen_mouse_10um"
    config_overrides: dict[str, Any] = field(default_factory=dict)
    alignments: dict[str, dict[str, Any]] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> BelljarProject:
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
