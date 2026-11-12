"""BrainGlobe Atlas API integration.

Provides a unified interface to brain atlases, replacing the custom
NRRD loading and structure_map.pkl with the BrainGlobe Atlas API.
Supports Allen Mouse, Waxholm Rat, and 30+ other atlases.
"""

from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast, get_args

from brainglobe_atlasapi.atlas_name import AtlasName

from belljar.atlas.intensity import rescale_reference

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

KNOWN_ATLAS_NAMES: tuple[str, ...] = get_args(AtlasName)


def known_atlas_name(name: str) -> AtlasName:
    """Narrow a configured atlas name to one BrainGlobe can serve."""
    if name not in KNOWN_ATLAS_NAMES:
        raise ValueError(
            f"Unknown atlas {name!r}; BrainGlobe knows {len(KNOWN_ATLAS_NAMES)} atlases "
            f"such as {', '.join(KNOWN_ATLAS_NAMES[:3])}"
        )
    return cast("AtlasName", name)


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
            self._atlas = BrainGlobeAtlas(known_atlas_name(self.atlas_name))
            logger.info(
                "Atlas loaded: %s (shape=%s, resolution=%sum)",
                self.atlas_name,
                self._atlas.reference.shape,
                self._atlas.resolution,
            )
        return self._atlas

    @cached_property
    def reference(self) -> NDArray[np.uint8]:
        """3D reference volume (grayscale atlas image) rescaled to uint8.

        Returns the default (STP) reference or an additional reference
        (e.g. Nissl) depending on ``self.reference_name``. The raw BrainGlobe
        volume is uint16; it is rescaled on first access and cached.
        """
        return rescale_reference(self._raw_reference())

    def _raw_reference(self) -> NDArray:
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
        """Shape of the atlas volume (Z, Y, X).

        Read from the annotation so shape queries never trigger the
        reference rescale.
        """
        return self.annotation.shape

    @property
    def resolution(self) -> tuple[float, ...]:
        """Voxel resolution in micrometers."""
        return self.atlas.resolution

    @cached_property
    def structure_map(self) -> dict[int, dict[str, Any]]:
        """Build a belljar-compatible structure map from BrainGlobe structures.

        Returns a dict mapping region_id (int) -> {name, acronym, color, id_path}.
        Keys are plain ints so lookups work with values read from JSON or from
        uint32 annotation arrays alike.
        """
        smap: dict[int, dict[str, Any]] = {}

        for structure_id, structure in self.atlas.structures.items():
            # BrainGlobe provides RGB color as a list
            rgb = structure.get("rgb_triplet", [128, 128, 128])
            color = tuple(rgb[:3]) if isinstance(rgb, (list, tuple)) else (128, 128, 128)

            # Build id_path from structure_id_path
            id_path = "/".join(str(x) for x in structure.get("structure_id_path", [structure_id]))

            smap[int(structure_id)] = {
                "name": structure.get("name", "Unknown"),
                "acronym": structure.get("acronym", "?"),
                "color": color,
                "id_path": id_path,
            }

        # Add "Lost in Warp" entry for region ID 0
        smap[0] = {
            "name": "Lost in Warp",
            "acronym": "LIW",
            "color": (0, 0, 0),
            "id_path": "0",
        }

        return smap

    @property
    def ap_range(self) -> tuple[int, int]:
        """Valid AP position range (0 to z_dim - 1)."""
        return (0, self.shape[0] - 1)

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
