"""Tests for registration quality metrics and preprocessing."""

import numpy as np
import pytest

from belljar.registration.preprocessing import (
    apply_layer_intensity_adjustments,
    resize_nearest_neighbor,
)


class TestLayerIntensityAdjustments:
    """Tests for vectorized layer intensity adjustment."""

    def test_adjustments_applied_correctly(self) -> None:
        """Layer 4 pixels should brighten, layer 5 should darken."""
        section = np.full((10, 10), 100, dtype=np.uint8)
        label = np.zeros((10, 10), dtype=np.uint32)
        label[:5, :] = 1  # Region 1 = layer 4
        label[5:, :] = 2  # Region 2 = layer 5

        structure_map = {
            np.uint32(1): {"name": "Primary visual area, layer 4"},
            np.uint32(2): {"name": "Primary visual area, layer 5"},
        }

        result = apply_layer_intensity_adjustments(
            section, label, structure_map, {"layer 4": 15, "layer 5": -7}
        )

        assert result[:5, :].mean() == 115  # 100 + 15
        assert result[5:, :].mean() == 93   # 100 - 7

    def test_clipping_at_boundaries(self) -> None:
        """Values should be clipped to [0, 255]."""
        section = np.full((4, 4), 250, dtype=np.uint8)
        label = np.ones((4, 4), dtype=np.uint32)
        structure_map = {np.uint32(1): {"name": "layer 4 region"}}

        result = apply_layer_intensity_adjustments(
            section, label, structure_map, {"layer 4": 15}
        )
        assert result.max() == 255  # Clipped, not 265

    def test_no_adjustments_for_unlisted_layers(self) -> None:
        """Regions not matching any layer should be unchanged."""
        section = np.full((4, 4), 100, dtype=np.uint8)
        label = np.ones((4, 4), dtype=np.uint32)
        structure_map = {np.uint32(1): {"name": "Some other region"}}

        result = apply_layer_intensity_adjustments(section, label, structure_map)
        np.testing.assert_array_equal(result, section)

    def test_empty_structure_map(self) -> None:
        """Empty structure map should leave image unchanged."""
        section = np.full((4, 4), 100, dtype=np.uint8)
        label = np.ones((4, 4), dtype=np.uint32)

        result = apply_layer_intensity_adjustments(section, label, {})
        np.testing.assert_array_equal(result, section)


class TestResizeNearestNeighbor:
    """Tests for nearest-neighbor resizing."""

    def test_preserves_label_values(self) -> None:
        """Resizing should only produce values present in the input (plus 0 from padding)."""
        image = np.array([[1, 2], [3, 4]], dtype=np.uint32)
        result = resize_nearest_neighbor(image, (4, 4))
        assert set(np.unique(result)).issubset({0, 1, 2, 3, 4})

    def test_output_shape(self) -> None:
        """Output should have the requested shape."""
        image = np.zeros((10, 10), dtype=np.uint8)
        result = resize_nearest_neighbor(image, (20, 20))
        assert result.shape == (20, 20)
