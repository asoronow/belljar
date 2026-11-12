"""Base classes for the Belljar processing pipeline.

Each processing step (max projection, sharpen, align, detect, count, collate)
inherits from PipelineStep, providing a uniform interface for validation,
execution, and progress reporting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Generic, Protocol, TypeVar

from belljar.atlas.provider import AtlasProvider
from belljar.pipeline.params import StepParams

if TYPE_CHECKING:
    from belljar.config import BelljarConfig
    from belljar.types import StepResult

P = TypeVar("P", bound=StepParams)


class ProgressCallback(Protocol):
    """Protocol for progress reporting callbacks."""

    def __call__(self, current: int, total: int, message: str) -> None: ...


class PipelineStep(ABC, Generic[P]):
    """Abstract base class for all pipeline steps.

    ``params_class`` names the model that callers build from CLI arguments or
    RPC params; ``run`` and ``validate_inputs`` receive an instance of it.
    """

    params_class: ClassVar[type[StepParams]]

    def __init__(
        self, config: BelljarConfig, atlas_provider: AtlasProvider | None = None
    ) -> None:
        self.config = config
        self._atlas_provider = atlas_provider

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this step."""
        ...

    @property
    def atlas_provider(self) -> AtlasProvider:
        """Atlas configured in ``config.atlas``, created on first use."""
        if self._atlas_provider is None:
            self._atlas_provider = AtlasProvider(
                self.config.atlas.atlas_name,
                reference_name=self.config.atlas.reference_name,
            )
        return self._atlas_provider

    def validate_inputs(self, params: P) -> list[str]:
        """Check filesystem preconditions that the params model cannot.

        Returns:
            List of validation error messages. Empty list means inputs are valid.
        """
        return []

    @abstractmethod
    def run(self, progress: ProgressCallback, params: P) -> StepResult:
        """Execute this pipeline step.

        Args:
            progress: Callback for reporting progress to the frontend.
            params: Validated step parameters.

        Returns:
            StepResult with success/failure, metrics, and any warnings.
        """
        ...


def _noop_progress(current: int, total: int, message: str) -> None:
    """Default no-op progress callback for testing."""
    pass
