"""Tests for atlas reference intensity rescaling."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from belljar.atlas.intensity import rescale_reference
from belljar.atlas.provider import AtlasProvider
from belljar.atlas.slicer import slice_3d_volume, slice_atlas_and_annotation


@pytest.fixture
def uint16_volume() -> tuple[np.ndarray, np.ndarray]:
    """A uint16 volume with a spherical tissue region spanning [0, 27000]."""
    shape = (60, 50, 40)
    z, y, x = np.ogrid[: shape[0], : shape[1], : shape[2]]
    distance = np.sqrt((z - 30) ** 2 + (y - 25) ** 2 + (x - 20) ** 2)
    tissue = distance < 18
    volume = np.zeros(shape, dtype=np.uint16)
    volume[tissue] = (27000 * (1 - distance[tissue] / 18)).astype(np.uint16) + 1
    return volume, tissue


def _provider_for(volume: np.ndarray) -> AtlasProvider:
    atlas = MagicMock()
    atlas.reference = volume
    atlas.annotation = np.ones(volume.shape, dtype=np.uint32)
    atlas.resolution = (10.0, 10.0, 10.0)
    provider = AtlasProvider("test_atlas", reference_name="default")
    provider._atlas = atlas
    return provider


class TestRescaleReference:
    def test_uint16_tissue_keeps_gray_levels(self, uint16_volume) -> None:
        volume, tissue = uint16_volume

        rescaled = rescale_reference(volume)

        assert rescaled.dtype == np.uint8
        assert rescaled.shape == volume.shape
        assert not rescaled[~tissue].any()
        assert len(np.unique(rescaled[tissue])) >= 50

    def test_subsampled_percentiles_match_full(self, uint16_volume) -> None:
        volume, _ = uint16_volume

        full = rescale_reference(volume, sample_voxels=volume.size)
        sampled = rescale_reference(volume, sample_voxels=5_000)

        assert np.abs(full.astype(np.int16) - sampled.astype(np.int16)).max() <= 3


class TestProviderReference:
    def test_reference_is_rescaled_uint8(self, uint16_volume) -> None:
        volume, tissue = uint16_volume
        provider = _provider_for(volume)

        reference = provider.reference

        assert reference.dtype == np.uint8
        assert len(np.unique(reference[tissue])) >= 50
        assert provider.reference is reference

    def test_shape_and_ap_range_skip_rescaling(self, uint16_volume) -> None:
        volume, _ = uint16_volume
        provider = _provider_for(volume)

        assert provider.shape == volume.shape
        assert provider.ap_range == (0, volume.shape[0] - 1)
        assert "reference" not in provider.__dict__

    def test_mid_plane_slice_keeps_gray_levels(self, uint16_volume) -> None:
        volume, _ = uint16_volume
        provider = _provider_for(volume)
        mid = volume.shape[0] // 2

        atlas_slice, _ = slice_atlas_and_annotation(
            provider.reference, provider.annotation, mid
        )

        assert atlas_slice.dtype == np.uint8
        assert len(np.unique(atlas_slice)) >= 50

    def test_generation_memmap_and_provider_slice_identical(
        self, uint16_volume, tmp_path
    ) -> None:
        """Data generation slices a memmap copy of provider.reference."""
        volume, _ = uint16_volume
        provider = _provider_for(volume)
        reference = provider.reference
        mid = volume.shape[0] // 2

        memmap_path = tmp_path / "ref.dat"
        memmap = np.memmap(
            str(memmap_path), dtype=reference.dtype, mode="w+", shape=reference.shape
        )
        memmap[:] = reference[:]
        memmap.flush()
        shared = np.memmap(
            str(memmap_path), dtype=str(reference.dtype), mode="r", shape=reference.shape
        )

        generation_slice = slice_3d_volume(shared, mid, 0.0, 0.0, 0.0, order=1)
        provider_slice, _ = slice_atlas_and_annotation(reference, provider.annotation, mid)

        np.testing.assert_array_equal(generation_slice, provider_slice)
        assert len(np.unique(generation_slice)) >= 50
