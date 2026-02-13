"""BrainGlobe Atlas API integration.

Provides a unified interface to brain atlases, replacing the custom
NRRD loading and structure_map.pkl with the BrainGlobe Atlas API.
Supports Allen Mouse, Waxholm Rat, and 30+ other atlases.
"""

from __future__ import annotations

import logging
from functools import cached_property
from typing import Any

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class AtlasProvider:
    """Wrapper around BrainGlobe Atlas API for belljar.

    Lazily loads atlas data on first access and provides a structure_map
    compatible with the existing belljar pipeline.

    Args:
        atlas_name: BrainGlobe atlas identifier (e.g., "allen_mouse_10um").
        reference_name: Which reference volume to use. "default" returns the
            primary reference (STP for Allen Mouse). Other values (e.g. "nissl")
            are looked up via BrainGlobe's additional_references mechanism.
    """

    def __init__(
        self,
        atlas_name: str = "allen_mouse_10um",
        reference_name: str = "default",
    ) -> None:
        self.atlas_name = atlas_name
        self.reference_name = reference_name
        self._atlas: Any = None

    @property
    def atlas(self) -> Any:
        """Lazily load the BrainGlobe atlas."""
        if self._atlas is None:
            from brainglobe_atlasapi import BrainGlobeAtlas

            logger.info("Loading atlas: %s", self.atlas_name)
            self._atlas = BrainGlobeAtlas(self.atlas_name)
            logger.info(
                "Atlas loaded: %s (shape=%s, resolution=%sum)",
                self.atlas_name,
                self._atlas.reference.shape,
                self._atlas.resolution,
            )
        return self._atlas

    @cached_property
    def reference(self) -> NDArray:
        """3D reference volume (grayscale atlas image).

        Returns the default (STP) reference or an additional reference
        (e.g. Nissl) depending on ``self.reference_name``.
        """
        if self.reference_name == "default":
            return self.atlas.reference

        additional = self.atlas.additional_references
        if self.reference_name not in additional:
            available = list(additional.keys()) if additional else []
            raise ValueError(
                f"Reference '{self.reference_name}' not available for "
                f"atlas '{self.atlas_name}'. Available additional references: {available}. "
                f"To add one, place '{self.reference_name}.tiff' in the atlas directory "
                f"and add it to metadata.json 'additional_references' list."
            )
        ref = additional[self.reference_name]
        logger.info("Using additional reference: %s", self.reference_name)
        return ref

    @cached_property
    def annotation(self) -> NDArray:
        """3D annotation volume (uint32 region IDs)."""
        return self.atlas.annotation

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the atlas volume (Z, Y, X)."""
        return self.reference.shape

    @property
    def resolution(self) -> tuple[float, ...]:
        """Voxel resolution in micrometers."""
        return self.atlas.resolution

    @cached_property
    def structure_map(self) -> dict[int, dict[str, Any]]:
        """Build a belljar-compatible structure map from BrainGlobe structures.

        Returns a dict mapping region_id -> {name, acronym, color, id_path}.
        Compatible with the existing pipeline's structure_map format.
        """
        smap: dict[int, dict[str, Any]] = {}

        for structure_id, structure in self.atlas.structures.items():
            # BrainGlobe provides RGB color as a list
            color = structure.get("rgb_triplet", [128, 128, 128])
            if isinstance(color, (list, tuple)):
                color = tuple(color[:3])
            else:
                color = (128, 128, 128)

            # Build id_path from structure_id_path
            id_path = "/".join(str(x) for x in structure.get("structure_id_path", [structure_id]))

            smap[np.uint32(structure_id)] = {
                "name": structure.get("name", "Unknown"),
                "acronym": structure.get("acronym", "?"),
                "color": color,
                "id_path": id_path,
            }

        # Add "Lost in Warp" entry for region ID 0
        smap[np.uint32(0)] = {
            "name": "Lost in Warp",
            "acronym": "LIW",
            "color": (0, 0, 0),
            "id_path": "0",
        }

        return smap

    @property
    def ap_range(self) -> tuple[int, int]:
        """Valid AP position range (0 to z_dim - 1)."""
        return (0, self.reference.shape[0] - 1)

    def get_hemisphere(self, hemisphere: str = "W") -> tuple[NDArray, NDArray]:
        """Get atlas volumes for a specific hemisphere.

        Args:
            hemisphere: "W" for whole, "L" for left hemisphere only.

        Returns:
            Tuple of (reference, annotation) volumes.
        """
        ref = self.reference
        ann = self.annotation
        if hemisphere == "L":
            mid = ref.shape[2] // 2
            ref = ref[:, :, :mid]
            ann = ann[:, :, :mid]
        return ref, ann
