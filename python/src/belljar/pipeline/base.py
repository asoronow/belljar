"""Base classes for the Belljar processing pipeline.

Each processing step (max projection, sharpen, align, detect, count, collate)
inherits from PipelineStep, providing a uniform interface for validation,
execution, and progress reporting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Protocol

from belljar.config import BelljarConfig
from belljar.types import StepResult


class ProgressCallback(Protocol):
    """Protocol for progress reporting callbacks."""

    def __call__(self, current: int, total: int, message: str) -> None: ...


class PipelineStep(ABC):
    """Abstract base class for all pipeline steps."""

    def __init__(self, config: BelljarConfig) -> None:
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this step."""
        ...

    @abstractmethod
    def validate_inputs(self, **kwargs: Any) -> list[str]:
        """Validate inputs before execution.

        Returns:
            List of validation error messages. Empty list means inputs are valid.
        """
        ...

    @abstractmethod
    def run(self, progress: ProgressCallback, **kwargs: Any) -> StepResult:
        """Execute this pipeline step.

        Args:
            progress: Callback for reporting progress to the frontend.
            **kwargs: Step-specific parameters.

        Returns:
            StepResult with success/failure, metrics, and any warnings.
        """
        ...


def _noop_progress(current: int, total: int, message: str) -> None:
    """Default no-op progress callback for testing."""
    pass
