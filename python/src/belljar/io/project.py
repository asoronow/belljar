"""Project file management.

Handles saving and loading belljar project state as JSON files,
replacing the opaque pickle-based serialization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from belljar.types import BelljarProject, SliceAlignment

logger = logging.getLogger(__name__)


def save_project(project: BelljarProject, path: Path) -> None:
    """Save a project to a JSON file.

    Args:
        project: Project state to save.
        path: Output file path (typically project.belljar).
    """
    project.save(path)
    logger.info("Project saved to %s", path)


def load_project(path: Path) -> BelljarProject:
    """Load a project from a JSON file.

    Args:
        path: Path to the project file.

    Returns:
        Loaded BelljarProject.
    """
    project = BelljarProject.load(path)
    logger.info("Project loaded from %s", project.name)
    return project


def save_annotation(annotation: np.ndarray, path: Path) -> None:
    """Save an annotation array as a numpy file.

    Args:
        annotation: 2D annotation array (uint32 region IDs).
        path: Output path (.npy extension).
    """
    np.save(path, annotation)


def load_annotation(path: Path) -> np.ndarray:
    """Load an annotation array from a numpy file.

    Args:
        path: Path to the .npy file.

    Returns:
        2D annotation array (uint32).
    """
    return np.load(path)


def migrate_pickle_alignment(pickle_path: Path) -> dict[str, SliceAlignment]:
    """Migrate a legacy alignment.pkl to the new format.

    Reads the old pickle-based alignment state and converts it to
    typed SliceAlignment objects that can be serialized as JSON.

    Args:
        pickle_path: Path to the legacy alignment.pkl file.

    Returns:
        Dict mapping section names to SliceAlignment objects.
    """
    import pickle

    with open(pickle_path, "rb") as f:
        old_data = pickle.load(f)

    alignments: dict[str, SliceAlignment] = {}
    for section_name, atlas_slice in old_data.items():
        alignments[section_name] = SliceAlignment(
            section_name=atlas_slice.section_name,
            ap_position=float(atlas_slice.ap_position),
            x_angle=float(atlas_slice.x_angle),
            y_angle=float(atlas_slice.y_angle),
            region=getattr(atlas_slice, "region", "A"),
            hemisphere=getattr(atlas_slice, "hemisphere", "W"),
            linked=getattr(atlas_slice, "linked", True),
        )

    logger.info("Migrated %d alignments from legacy pickle format", len(alignments))
    return alignments
