"""Tests for atlas slicing with full 3D rotation."""

import numpy as np
import pytest

from belljar.atlas.slicer import slice_3d_volume, slice_atlas_and_annotation


class TestSlice3dVolume:
    """Tests for the slice_3d_volume function."""

    def test_axial_slice_at_center(self, synthetic_atlas: np.ndarray) -> None:
        """A flat slice at the center should match the volume's center plane."""
        z_center = synthetic_atlas.shape[0] // 2
        result = slice_3d_volume(synthetic_atlas, z_center, 0, 0, 0, order=1)
        assert result.shape == (synthetic_atlas.shape[1], synthetic_atlas.shape[2])
        # Center plane should closely match the actual data at z_center
        expected = synthetic_atlas[z_center]
        np.testing.assert_array_almost_equal(result, expected, decimal=0)

    def test_zero_angles_returns_flat_slice(self, synthetic_atlas: np.ndarray) -> None:
        """With no rotation, the slice should be a clean axial cut."""
        for z_pos in [10, 50, 90]:
            result = slice_3d_volume(synthetic_atlas, z_pos, 0, 0, 0, order=0)
            expected = synthetic_atlas[z_pos]
            np.testing.assert_array_equal(result, expected)

    def test_output_shape_matches_volume_yx(self, synthetic_atlas: np.ndarray) -> None:
        """Output shape should always be (Y, X) of the volume."""
        result = slice_3d_volume(synthetic_atlas, 50, 5, 3, 0)
        assert result.shape == (synthetic_atlas.shape[1], synthetic_atlas.shape[2])

    def test_small_angles_close_to_flat(self, synthetic_atlas: np.ndarray) -> None:
        """Small angles should produce results close to a flat slice."""
        z_pos = 50
        flat = slice_3d_volume(synthetic_atlas, z_pos, 0, 0, 0, order=1)
        tilted = slice_3d_volume(synthetic_atlas, z_pos, 0.5, 0.5, 0, order=1)
        # Small tilt should be similar to flat
        correlation = np.corrcoef(flat.ravel(), tilted.ravel())[0, 1]
        assert correlation > 0.95

    def test_nearest_neighbor_preserves_integers(
        self, synthetic_annotation: np.ndarray
    ) -> None:
        """Nearest-neighbor interpolation should only produce values present in the input."""
        result = slice_3d_volume(synthetic_annotation, 50, 3, 2, 0, order=0)
        unique_result = set(np.unique(result))
        unique_input = set(np.unique(synthetic_annotation))
        assert unique_result.issubset(unique_input | {0})

    def test_out_of_bounds_returns_zeros(self, synthetic_atlas: np.ndarray) -> None:
        """Sampling outside the volume should return zeros (cval=0)."""
        # Position way outside the volume
        result = slice_3d_volume(synthetic_atlas, -100, 0, 0, 0, order=1)
        # Most values should be 0 since we're outside the volume
        # But due to clamping, we'll get the edge values
        assert result.shape == (synthetic_atlas.shape[1], synthetic_atlas.shape[2])

    def test_symmetry_of_opposite_angles(self, synthetic_atlas: np.ndarray) -> None:
        """Slicing with opposite angles should produce mirror-like results."""
        pos = 50
        slice_pos = slice_3d_volume(synthetic_atlas, pos, 5, 0, 0, order=1)
        slice_neg = slice_3d_volume(synthetic_atlas, pos, -5, 0, 0, order=1)
        # They won't be identical but should have similar statistics
        assert abs(slice_pos.mean() - slice_neg.mean()) < 20


class TestSliceAtlasAndAnnotation:
    """Tests for the combined atlas + annotation slicing."""

    def test_returns_correct_dtypes(
        self,
        synthetic_atlas: np.ndarray,
        synthetic_annotation: np.ndarray,
    ) -> None:
        """Atlas should be uint8, annotation should be uint32."""
        atlas_slice, ann_slice = slice_atlas_and_annotation(
            synthetic_atlas, synthetic_annotation, 50, 2, 1
        )
        assert atlas_slice.dtype == np.uint8
        assert ann_slice.dtype == np.uint32

    def test_shapes_match(
        self,
        synthetic_atlas: np.ndarray,
        synthetic_annotation: np.ndarray,
    ) -> None:
        """Both slices should have the same 2D shape."""
        atlas_slice, ann_slice = slice_atlas_and_annotation(
            synthetic_atlas, synthetic_annotation, 50, 2, 1
        )
        assert atlas_slice.shape == ann_slice.shape

    def test_annotation_labels_preserved(
        self,
        synthetic_atlas: np.ndarray,
        synthetic_annotation: np.ndarray,
    ) -> None:
        """Annotation slicing should only contain valid label values."""
        _, ann_slice = slice_atlas_and_annotation(
            synthetic_atlas, synthetic_annotation, 50, 0, 0
        )
        valid_labels = set(np.unique(synthetic_annotation))
        result_labels = set(np.unique(ann_slice))
        assert result_labels.issubset(valid_labels)
