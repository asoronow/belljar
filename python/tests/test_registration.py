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


def _blob_image(size: int) -> np.ndarray:
    """Blurred discs of varying intensity, enough structure to anchor registration."""
    import cv2

    yy, xx = np.mgrid[0:size, 0:size]
    image = np.zeros((size, size), dtype=np.float32)
    scale = size / 128
    for cx, cy, radius, value in [
        (64, 64, 30, 200),
        (50, 60, 8, 60),
        (78, 70, 6, 120),
        (64, 40, 5, 250),
        (40, 80, 7, 180),
    ]:
        disc = (xx - cx * scale) ** 2 + (yy - cy * scale) ** 2 < (radius * scale) ** 2
        image[disc] = value
    return cv2.GaussianBlur(image, (5, 5), 1.5)


def _shift(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = np.zeros_like(image)
    h, w = image.shape
    shifted[dy:, dx:] = image[: h - dy, : w - dx]
    return shifted


class TestComputeRegistrationQuality:
    """Tests for compute_registration_quality."""

    def test_identical_images_score_high(self) -> None:
        import SimpleITK as sitk

        from belljar.registration.metrics import compute_registration_quality

        image = sitk.GetImageFromArray(_blob_image(64))
        metrics = compute_registration_quality(image, image)

        assert metrics.mutual_information > 0
        assert metrics.normalized_cross_correlation == pytest.approx(1.0, abs=1e-6)

    def test_shifted_copy_scores_lower(self) -> None:
        import SimpleITK as sitk

        from belljar.registration.metrics import compute_registration_quality

        array = _blob_image(64)
        fixed = sitk.GetImageFromArray(array)
        shifted = sitk.GetImageFromArray(_shift(array, 10, 0))

        aligned = compute_registration_quality(fixed, fixed)
        misaligned = compute_registration_quality(fixed, shifted)

        assert misaligned.mutual_information < aligned.mutual_information
        assert misaligned.normalized_cross_correlation < aligned.normalized_cross_correlation


class TestMultimodalRegistrationKnownAnswer:
    """The rigid stage must recover a known translation."""

    @pytest.mark.parametrize("dx,dy", [(10, 0), (6, -8)])
    def test_recovers_translation(self, dx: int, dy: int) -> None:
        import SimpleITK as sitk

        from belljar.config import RegistrationConfig
        from belljar.registration.classical import multimodal_registration

        array = _blob_image(128)
        moving = _shift(array, max(dx, 0), max(dy, 0))
        fixed_array = _shift(array, max(-dx, 0), max(-dy, 0))
        fixed = sitk.GetImageFromArray(fixed_array)
        moving_image = sitk.GetImageFromArray(moving)

        config = RegistrationConfig(bspline_iterations=5)
        composite = multimodal_registration(fixed, moving_image, config)

        rigid = sitk.Euler2DTransform(composite.GetNthTransform(0).Downcast())
        tx, ty = rigid.GetTranslation()
        assert tx == pytest.approx(dx, abs=1.0)
        assert ty == pytest.approx(dy, abs=1.0)
        assert abs(rigid.GetAngle()) < np.radians(1.0)
