"""Tests for the training data generation module."""

import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

from belljar.config import AtlasConfig, DataGenerationConfig, EstimationConfig
from belljar.atlas.provider import AtlasProvider
from belljar.estimation.data_generation import (
    clahe_normalize,
    compute_anchoring_from_rotation,
    apply_domain_randomization,
    _elastic_deform,
    _worker_generate_batch,
    generate_single_sample,
    simulate_stain,
    STAIN_PROFILES,
)
from belljar.estimation.dataset import AngledAtlasDataset, TissueDataset
from belljar.estimation.predictor import anchoring_to_legacy, _preprocess_image


@pytest.fixture
def small_volume() -> np.ndarray:
    """A small synthetic atlas volume for testing."""
    rng = np.random.default_rng(0)
    vol = rng.integers(0, 255, (100, 64, 64), dtype=np.uint8)
    return vol


@pytest.fixture
def config() -> DataGenerationConfig:
    return DataGenerationConfig(
        z_range=(10, 90),
        x_angle_range=(-10.0, 10.0),
        y_angle_range=(-10.0, 10.0),
        z_angle_range=(-5.0, 5.0),
    )


class TestCLAHENormalize:
    def test_output_is_uint8(self) -> None:
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        result = clahe_normalize(img)
        assert result.dtype == np.uint8

    def test_preserves_shape(self) -> None:
        img = np.random.randint(0, 255, (100, 80), dtype=np.uint8)
        result = clahe_normalize(img)
        assert result.shape == img.shape

    def test_handles_uint16(self) -> None:
        img = np.random.randint(0, 65535, (64, 64), dtype=np.uint16)
        result = clahe_normalize(img)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64)

    def test_handles_float(self) -> None:
        img = np.random.rand(64, 64).astype(np.float32)
        result = clahe_normalize(img)
        assert result.dtype == np.uint8

    def test_no_nans(self) -> None:
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        result = clahe_normalize(img)
        assert not np.any(np.isnan(result.astype(np.float32)))


class TestComputeAnchoringFromRotation:
    def test_output_length(self) -> None:
        result = compute_anchoring_from_rotation(50.0, 0.0, 0.0, 0.0, (100, 64, 64))
        assert len(result) == 9

    def test_zero_angles_identity(self) -> None:
        """With zero angles, u should be [0,0,1] and v should be [0,1,0]."""
        result = compute_anchoring_from_rotation(50.0, 0.0, 0.0, 0.0, (100, 64, 64))
        ox, oy, oz, ux, uy, uz, vx, vy, vz = result
        assert abs(ox - 0.5) < 1e-6
        assert abs(oy - 0.5) < 1e-6
        assert abs(oz - 0.5) < 1e-6  # 50/100
        # u = width = rotated [0,0,1] in [z,y,x] -> swapped to [x,y,z] = [1,0,0]
        assert abs(ux - 1.0) < 1e-6
        assert abs(uy - 0.0) < 1e-6
        assert abs(uz - 0.0) < 1e-6
        # v = height = rotated [0,1,0] in [z,y,x] -> swapped to [x,y,z] = [0,1,0]
        assert abs(vx - 0.0) < 1e-6
        assert abs(vy - 1.0) < 1e-6
        assert abs(vz - 0.0) < 1e-6

    def test_ap_position_normalization(self) -> None:
        """AP position should be normalized by ap_range."""
        result = compute_anchoring_from_rotation(
            662.0, 0.0, 0.0, 0.0, (1320, 800, 1140), ap_range=(0.0, 1320.0)
        )
        oz = result[2]
        assert abs(oz - 0.5015) < 0.01  # ~662/1320

    def test_roundtrip_z_position(self) -> None:
        """Z position should roundtrip accurately through anchoring_to_legacy."""
        z_pos = 500.0
        ap_range = (0.0, 1324.0)
        anchoring = compute_anchoring_from_rotation(
            z_pos, 3.0, -2.0, 0.0, (1324, 800, 1140), ap_range=ap_range
        )
        z_rec, _, _ = anchoring_to_legacy(anchoring, ap_range=ap_range)
        assert abs(z_rec - z_pos) < 2.0

    def test_roundtrip_zero_angles(self) -> None:
        """With zero angles, legacy roundtrip should be exact."""
        z_pos = 662.0
        ap_range = (0.0, 1324.0)
        anchoring = compute_anchoring_from_rotation(
            z_pos, 0.0, 0.0, 0.0, (1324, 800, 1140), ap_range=ap_range
        )
        z_rec, x_rec, y_rec = anchoring_to_legacy(anchoring, ap_range=ap_range)
        assert abs(z_rec - z_pos) < 1.0
        assert abs(x_rec) < 0.1
        assert abs(y_rec) < 0.1

    def test_different_z_gives_different_oz(self) -> None:
        """Different z positions should produce different oz values."""
        a1 = compute_anchoring_from_rotation(30.0, 0.0, 0.0, 0.0, (100, 64, 64))
        a2 = compute_anchoring_from_rotation(70.0, 0.0, 0.0, 0.0, (100, 64, 64))
        assert a1[2] != a2[2]

    def test_nonzero_angles_change_uv(self) -> None:
        """Nonzero angles should rotate u and v from their identity values."""
        zero = compute_anchoring_from_rotation(50.0, 0.0, 0.0, 0.0, (100, 64, 64))
        tilted = compute_anchoring_from_rotation(50.0, 5.0, 3.0, 0.0, (100, 64, 64))
        # u and v vectors should differ
        assert zero[3:6] != tilted[3:6]
        assert zero[6:9] != tilted[6:9]


class TestDomainRandomization:
    def test_output_dtype_and_range(self) -> None:
        rng = np.random.default_rng(42)
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        result = apply_domain_randomization(img, rng)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_preserves_shape(self) -> None:
        rng = np.random.default_rng(42)
        img = np.random.randint(0, 255, (100, 80), dtype=np.uint8)
        result = apply_domain_randomization(img, rng)
        assert result.shape == img.shape

    def test_different_seeds_produce_different_results(self) -> None:
        img = np.random.randint(50, 200, (64, 64), dtype=np.uint8)
        r1 = apply_domain_randomization(img, np.random.default_rng(0))
        r2 = apply_domain_randomization(img, np.random.default_rng(999))
        assert not np.array_equal(r1, r2)

    def test_no_nans_or_infs(self) -> None:
        rng = np.random.default_rng(42)
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        for seed in range(20):
            result = apply_domain_randomization(img, np.random.default_rng(seed))
            result_f = result.astype(np.float32)
            assert not np.any(np.isnan(result_f))
            assert not np.any(np.isinf(result_f))


class TestElasticDeform:
    def test_output_shape(self) -> None:
        rng = np.random.default_rng(0)
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        result = _elastic_deform(img, rng)
        assert result.shape == img.shape

    def test_output_dtype(self) -> None:
        rng = np.random.default_rng(0)
        img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        result = _elastic_deform(img, rng)
        assert result.dtype == np.uint8


class TestGenerateSingleSample:
    def test_output_types(self, small_volume: np.ndarray, config: DataGenerationConfig) -> None:
        rng = np.random.default_rng(42)
        image, anchoring, metadata = generate_single_sample(small_volume, rng, config)
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        assert isinstance(anchoring, list)
        assert len(anchoring) == 9
        assert isinstance(metadata, dict)

    def test_image_shape(self, small_volume: np.ndarray, config: DataGenerationConfig) -> None:
        rng = np.random.default_rng(42)
        image, _, _ = generate_single_sample(small_volume, rng, config)
        assert image.shape == (256, 256)

    def test_metadata_keys(self, small_volume: np.ndarray, config: DataGenerationConfig) -> None:
        rng = np.random.default_rng(42)
        _, _, metadata = generate_single_sample(small_volume, rng, config)
        expected_keys = {"pos", "x_angle", "y_angle", "z_angle", "anchoring", "is_hemi"}
        assert expected_keys.issubset(metadata.keys())

    def test_angles_within_range(self, small_volume: np.ndarray, config: DataGenerationConfig) -> None:
        rng = np.random.default_rng(42)
        for seed in range(10):
            _, _, meta = generate_single_sample(small_volume, np.random.default_rng(seed), config)
            assert config.x_angle_range[0] <= meta["x_angle"] <= config.x_angle_range[1]
            assert config.y_angle_range[0] <= meta["y_angle"] <= config.y_angle_range[1]
            assert config.z_angle_range[0] <= meta["z_angle"] <= config.z_angle_range[1]

    def test_pos_within_range(self, small_volume: np.ndarray, config: DataGenerationConfig) -> None:
        rng = np.random.default_rng(42)
        for seed in range(10):
            _, _, meta = generate_single_sample(small_volume, np.random.default_rng(seed), config)
            assert config.z_range[0] <= meta["pos"] <= config.z_range[1]

    def test_reproducible_with_same_seed(
        self, small_volume: np.ndarray, config: DataGenerationConfig
    ) -> None:
        img1, a1, _ = generate_single_sample(small_volume, np.random.default_rng(42), config)
        img2, a2, _ = generate_single_sample(small_volume, np.random.default_rng(42), config)
        np.testing.assert_array_equal(img1, img2)
        assert a1 == a2


class TestDatasetWithNewMetadata:
    """Test that AngledAtlasDataset loads the new anchoring metadata format."""

    def test_loads_anchoring_directly(self) -> None:
        """When metadata has 'anchoring' key, use it directly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)
            # Create a small PNG
            img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            cv2.imwrite(str(data_path / "sample_001.png"), img)

            # Create metadata with precomputed anchoring
            anchoring_val = [0.5, 0.5, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            metadata = {
                "sample_001": {
                    "pos": 400.0,
                    "x_angle": 2.0,
                    "y_angle": -1.0,
                    "anchoring": anchoring_val,
                }
            }
            with open(data_path / "metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)

            from torchvision import transforms

            tx = transforms.Compose([transforms.ToTensor()])
            ds = AngledAtlasDataset(data_path, transform=tx, output_format="anchoring")
            image_tensor, label = ds[0]

            assert label.shape == (9,)
            np.testing.assert_allclose(label.numpy(), anchoring_val, rtol=1e-5)

    def test_falls_back_to_legacy_conversion(self) -> None:
        """Without 'anchoring' key, fall back to legacy_to_anchoring conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)
            img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            cv2.imwrite(str(data_path / "sample_001.png"), img)

            metadata = {
                "sample_001": {"pos": 500.0, "x_angle": 0.0, "y_angle": 0.0}
            }
            with open(data_path / "metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)

            from torchvision import transforms

            tx = transforms.Compose([transforms.ToTensor()])
            ds = AngledAtlasDataset(data_path, transform=tx, output_format="anchoring")
            _, label = ds[0]
            assert label.shape == (9,)

    def test_clahe_preprocessing(self) -> None:
        """Dataset with preprocessing='clahe' should apply CLAHE."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir)
            img = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
            cv2.imwrite(str(data_path / "sample_001.png"), img)

            metadata = {"sample_001": {"pos": 100.0, "x_angle": 0.0, "y_angle": 0.0, "anchoring": [0.5]*9}}
            with open(data_path / "metadata.pkl", "wb") as f:
                pickle.dump(metadata, f)

            from torchvision import transforms

            tx = transforms.Compose([transforms.ToTensor()])
            ds_clahe = AngledAtlasDataset(data_path, transform=tx, preprocessing="clahe")
            ds_none = AngledAtlasDataset(data_path, transform=tx, preprocessing="none")

            img_clahe, _ = ds_clahe[0]
            img_none, _ = ds_none[0]
            # CLAHE output should differ from raw
            assert not torch.allclose(img_clahe, img_none)


class TestPreprocessImageConfigurable:
    """Test that _preprocess_image respects config.preprocessing."""

    def test_clahe_mode(self) -> None:
        config = EstimationConfig(preprocessing="clahe")
        img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        tensor = _preprocess_image(img, config)
        assert tensor.shape == (1, 1, 256, 256)

    def test_sobel_mode(self) -> None:
        config = EstimationConfig(preprocessing="sobel")
        img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        tensor = _preprocess_image(img, config)
        assert tensor.shape == (1, 1, 256, 256)

    def test_none_mode(self) -> None:
        config = EstimationConfig(preprocessing="none")
        img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        tensor = _preprocess_image(img, config)
        assert tensor.shape == (1, 1, 256, 256)

    def test_different_modes_produce_different_output(self) -> None:
        img = np.random.randint(50, 200, (128, 128), dtype=np.uint8)
        t_clahe = _preprocess_image(img, EstimationConfig(preprocessing="clahe"))
        t_sobel = _preprocess_image(img, EstimationConfig(preprocessing="sobel"))
        t_none = _preprocess_image(img, EstimationConfig(preprocessing="none"))
        assert not torch.allclose(t_clahe, t_sobel)
        assert not torch.allclose(t_clahe, t_none)


class TestAtlasReferenceSelection:
    """Test that AtlasProvider supports reference modality selection."""

    def test_default_config(self) -> None:
        """AtlasConfig defaults to 'default' reference."""
        config = AtlasConfig()
        assert config.reference_name == "default"

    def test_nissl_config(self) -> None:
        """AtlasConfig accepts 'nissl' reference."""
        config = AtlasConfig(reference_name="nissl")
        assert config.reference_name == "nissl"

    def test_provider_stores_reference_name(self) -> None:
        """AtlasProvider stores reference_name parameter."""
        provider = AtlasProvider("allen_mouse_10um", reference_name="nissl")
        assert provider.reference_name == "nissl"

    def test_provider_default_reference(self) -> None:
        """Default reference_name is 'default'."""
        provider = AtlasProvider("allen_mouse_10um")
        assert provider.reference_name == "default"

    def test_provider_default_uses_primary_reference(self) -> None:
        """With reference_name='default', provider.reference uses atlas.reference."""
        fake_ref = np.zeros((10, 8, 8), dtype=np.uint8)
        mock_atlas = MagicMock()
        mock_atlas.reference = fake_ref
        mock_atlas.resolution = (10.0, 10.0, 10.0)

        provider = AtlasProvider("test_atlas", reference_name="default")
        provider._atlas = mock_atlas

        ref = provider.reference
        np.testing.assert_array_equal(ref, fake_ref)

    def test_provider_additional_reference(self) -> None:
        """With reference_name='nissl', provider.reference uses additional_references."""
        fake_stp = np.zeros((10, 8, 8), dtype=np.uint8)
        fake_nissl = np.ones((10, 8, 8), dtype=np.uint8) * 128

        mock_atlas = MagicMock()
        mock_atlas.reference = fake_stp
        mock_atlas.additional_references = {"nissl": fake_nissl}
        mock_atlas.resolution = (10.0, 10.0, 10.0)

        provider = AtlasProvider("test_atlas", reference_name="nissl")
        provider._atlas = mock_atlas

        ref = provider.reference
        np.testing.assert_array_equal(ref, fake_nissl)

    def test_provider_missing_reference_raises(self) -> None:
        """Requesting an unavailable reference raises ValueError."""
        mock_atlas = MagicMock()
        mock_atlas.additional_references = {}
        mock_atlas.resolution = (10.0, 10.0, 10.0)

        provider = AtlasProvider("test_atlas", reference_name="nissl")
        provider._atlas = mock_atlas

        with pytest.raises(ValueError, match="not available"):
            _ = provider.reference

    def test_generate_sample_with_different_volumes(
        self, config: DataGenerationConfig
    ) -> None:
        """generate_single_sample produces different output for different volumes."""
        rng_seed = 42
        vol_a = np.random.default_rng(0).integers(0, 255, (100, 64, 64), dtype=np.uint8)
        vol_b = np.random.default_rng(1).integers(0, 255, (100, 64, 64), dtype=np.uint8)

        img_a, _, _ = generate_single_sample(vol_a, np.random.default_rng(rng_seed), config)
        img_b, _, _ = generate_single_sample(vol_b, np.random.default_rng(rng_seed), config)

        # Same angles/position but different source volumes should produce different images
        assert not np.array_equal(img_a, img_b)


@pytest.fixture
def synthetic_nissl() -> np.ndarray:
    """A 256x256 synthetic Nissl-like coronal section with continuous gradients.

    Models the intensity structure of a real Nissl atlas slice:
    - Light background (~230) outside an elliptical brain region
    - Cortical layers as concentric intensity bands (140-200)
    - Dark white matter region (~70)
    - Very dark ventricle (~15)
    - Smooth Gaussian-blurred transitions between regions
    """
    img = np.full((256, 256), 230, dtype=np.float32)
    yy, xx = np.mgrid[:256, :256]

    # Elliptical brain region
    cy, cx = 130, 128
    brain_dist = ((yy - cy) / 95.0) ** 2 + ((xx - cx) / 110.0) ** 2
    brain_mask = brain_dist < 1.0

    # Cortical layers: concentric bands with varying intensity (outer=lighter, inner=darker)
    # Simulates layers 1-6 going from pial surface inward
    cortex_depth = np.clip(1.0 - brain_dist, 0, 1)  # 0 at edge, 1 at center
    cortex_intensity = 200 - cortex_depth * 60  # 200 (outer) → 140 (inner)
    img[brain_mask] = cortex_intensity[brain_mask]

    # White matter: inner elliptical ring (dark, ~70)
    wm_dist = ((yy - cy) / 50.0) ** 2 + ((xx - cx) / 60.0) ** 2
    wm_mask = (wm_dist < 1.0) & (wm_dist > 0.4)
    img[wm_mask] = 70

    # Ventricle: small dark region at center (~15)
    vent_dist = ((yy - cy) / 20.0) ** 2 + ((xx - (cx + 5)) / 15.0) ** 2
    vent_mask = vent_dist < 1.0
    img[vent_mask] = 15

    # Hippocampus-like bright region offset from center
    hipp_dist = ((yy - (cy + 30)) / 25.0) ** 2 + ((xx - (cx - 20)) / 35.0) ** 2
    hipp_mask = (hipp_dist < 1.0) & brain_mask
    img[hipp_mask] = 185

    # Smooth all transitions
    img = cv2.GaussianBlur(img, (7, 7), 0)

    return np.clip(img, 0, 255).astype(np.uint8)


class TestStainSimulation:
    """Tests for stain-mode-aware intensity simulation with structure preservation."""

    def test_internal_structure_preserved(self, synthetic_nissl: np.ndarray) -> None:
        """Brain region must retain internal intensity variation (not a flat blob)."""
        yy, xx = np.mgrid[:256, :256]
        brain_mask = ((yy - 130) / 95.0) ** 2 + ((xx - 128) / 110.0) ** 2 < 1.0

        for mode in STAIN_PROFILES:
            rng = np.random.default_rng(42)
            result = simulate_stain(
                synthetic_nissl.astype(np.float32), rng, stain_weights={mode: 1.0}
            )
            brain_std = np.std(result[brain_mask])
            assert brain_std > 15, (
                f"Stain '{mode}': brain region is a flat blob (std={brain_std:.1f})"
            )

    def test_gradient_magnitude_preserved(self, synthetic_nissl: np.ndarray) -> None:
        """Sobel gradient magnitude of output should be >=40% of input."""
        input_f = synthetic_nissl.astype(np.float32)
        gx_in = cv2.Sobel(input_f, cv2.CV_32F, 1, 0, ksize=3)
        gy_in = cv2.Sobel(input_f, cv2.CV_32F, 0, 1, ksize=3)
        grad_in = float(np.mean(np.sqrt(gx_in**2 + gy_in**2)))

        for mode in STAIN_PROFILES:
            rng = np.random.default_rng(42)
            result = simulate_stain(input_f, rng, stain_weights={mode: 1.0})
            gx_out = cv2.Sobel(result, cv2.CV_32F, 1, 0, ksize=3)
            gy_out = cv2.Sobel(result, cv2.CV_32F, 0, 1, ksize=3)
            grad_out = float(np.mean(np.sqrt(gx_out**2 + gy_out**2)))

            assert grad_out >= grad_in * 0.40, (
                f"Stain '{mode}': gradient magnitude too low "
                f"({grad_out:.1f} vs {grad_in:.1f} input)"
            )

    def test_regional_contrast_preserved(self, synthetic_nissl: np.ndarray) -> None:
        """Three known regions should have distinct mean intensities after transform."""
        # Cortex outer (bright in Nissl), white matter (dark), ventricle (very dark)
        yy, xx = np.mgrid[:256, :256]
        cortex_mask = (
            (((yy - 130) / 95.0) ** 2 + ((xx - 128) / 110.0) ** 2 < 1.0)
            & (((yy - 130) / 95.0) ** 2 + ((xx - 128) / 110.0) ** 2 > 0.7)
        )
        wm_mask = (
            (((yy - 130) / 50.0) ** 2 + ((xx - 128) / 60.0) ** 2 < 1.0)
            & (((yy - 130) / 50.0) ** 2 + ((xx - 128) / 60.0) ** 2 > 0.4)
        )
        vent_mask = ((yy - 130) / 20.0) ** 2 + ((xx - 133) / 15.0) ** 2 < 1.0

        for mode in STAIN_PROFILES:
            rng = np.random.default_rng(42)
            result = simulate_stain(
                synthetic_nissl.astype(np.float32), rng, stain_weights={mode: 1.0}
            )
            m_cortex = float(np.mean(result[cortex_mask]))
            m_wm = float(np.mean(result[wm_mask]))
            m_vent = float(np.mean(result[vent_mask]))

            # All three should be distinct (at least 10 apart)
            diffs = [abs(m_cortex - m_wm), abs(m_cortex - m_vent), abs(m_wm - m_vent)]
            assert all(d > 10 for d in diffs), (
                f"Stain '{mode}': regions not distinct "
                f"(cortex={m_cortex:.0f}, wm={m_wm:.0f}, vent={m_vent:.0f})"
            )

    def test_absorption_polarity(self, synthetic_nissl: np.ndarray) -> None:
        """Absorption stains: brain interior should be darker than background."""
        yy, xx = np.mgrid[:256, :256]
        brain_mask = ((yy - 130) / 95.0) ** 2 + ((xx - 128) / 110.0) ** 2 < 1.0
        bg_mask = ((yy - 130) / 95.0) ** 2 + ((xx - 128) / 110.0) ** 2 > 1.2

        for mode in ("nissl", "ache", "he"):
            rng = np.random.default_rng(42)
            result = simulate_stain(
                synthetic_nissl.astype(np.float32), rng, stain_weights={mode: 1.0}
            )
            brain_mean = float(np.mean(result[brain_mask]))
            bg_mean = float(np.mean(result[bg_mask]))
            assert brain_mean < bg_mean, (
                f"Absorption stain '{mode}': brain ({brain_mean:.0f}) should be "
                f"darker than background ({bg_mean:.0f})"
            )

    def test_fluorescence_polarity(self, synthetic_nissl: np.ndarray) -> None:
        """Fluorescence stains: brain interior should be brighter than background."""
        yy, xx = np.mgrid[:256, :256]
        brain_mask = ((yy - 130) / 95.0) ** 2 + ((xx - 128) / 110.0) ** 2 < 1.0
        bg_mask = ((yy - 130) / 95.0) ** 2 + ((xx - 128) / 110.0) ** 2 > 1.2

        for mode in ("dapi", "fluorescence"):
            rng = np.random.default_rng(42)
            result = simulate_stain(
                synthetic_nissl.astype(np.float32), rng, stain_weights={mode: 1.0}
            )
            brain_mean = float(np.mean(result[brain_mask]))
            bg_mean = float(np.mean(result[bg_mask]))
            assert brain_mean > bg_mean, (
                f"Fluorescence stain '{mode}': brain ({brain_mean:.0f}) should be "
                f"brighter than background ({bg_mean:.0f})"
            )

    def test_no_crushed_images(self, synthetic_nissl: np.ndarray) -> None:
        """No stain mode should produce all-black or all-white images."""
        for mode in STAIN_PROFILES:
            for seed in range(20):
                rng = np.random.default_rng(seed)
                result = simulate_stain(
                    synthetic_nissl.astype(np.float32), rng, stain_weights={mode: 1.0}
                )
                mean_val = np.mean(result)
                assert mean_val > 10, (
                    f"Stain '{mode}' seed={seed}: nearly all-black (mean={mean_val:.1f})"
                )
                assert mean_val < 245, (
                    f"Stain '{mode}' seed={seed}: nearly all-white (mean={mean_val:.1f})"
                )

    def test_stain_weights_with_ache(self, synthetic_nissl: np.ndarray) -> None:
        """Default weights should select each mode at approximately expected rates."""
        weights = {"nissl": 0.30, "dapi": 0.20, "ache": 0.15, "he": 0.15, "fluorescence": 0.20}
        n_trials = 500
        dark_bg_count = 0
        for seed in range(n_trials):
            rng = np.random.default_rng(seed)
            result = simulate_stain(
                synthetic_nissl.astype(np.float32), rng, stain_weights=weights
            )
            if np.mean(result) < 128:
                dark_bg_count += 1

        # Expected dark-bg fraction: dapi(0.20) + fluorescence(0.20) = 0.40
        dark_frac = dark_bg_count / n_trials
        assert 0.25 < dark_frac < 0.55, (
            f"Expected ~40% dark-background images, got {dark_frac:.2%}"
        )

    def test_modes_produce_distinct_outputs(self, synthetic_nissl: np.ndarray) -> None:
        """Absorption vs fluorescence modes should produce different mean intensities."""
        mode_means: dict[str, float] = {}
        for mode in STAIN_PROFILES:
            means = []
            for seed in range(20):
                rng = np.random.default_rng(seed)
                result = simulate_stain(
                    synthetic_nissl.astype(np.float32), rng, stain_weights={mode: 1.0}
                )
                means.append(float(np.mean(result)))
            mode_means[mode] = np.mean(means)

        absorption_mean = (mode_means["nissl"] + mode_means["he"]) / 2
        fluorescence_mean = (mode_means["dapi"] + mode_means["fluorescence"]) / 2
        assert abs(absorption_mean - fluorescence_mean) > 25, (
            f"Absorption mean ({absorption_mean:.1f}) and fluorescence mean "
            f"({fluorescence_mean:.1f}) are too similar"
        )

    def test_output_dtype_and_range(self, synthetic_nissl: np.ndarray) -> None:
        """simulate_stain output should be float32 in [0, 255]."""
        for mode in STAIN_PROFILES:
            rng = np.random.default_rng(42)
            result = simulate_stain(
                synthetic_nissl.astype(np.float32), rng, stain_weights={mode: 1.0}
            )
            assert result.dtype == np.float32
            assert result.min() >= 0.0
            assert result.max() <= 255.0

    def test_domain_randomization_integration(self, synthetic_nissl: np.ndarray) -> None:
        """apply_domain_randomization should still produce valid output."""
        for seed in range(20):
            rng = np.random.default_rng(seed)
            result = apply_domain_randomization(synthetic_nissl, rng)
            assert result.dtype == np.uint8
            assert result.min() >= 0
            assert result.max() <= 255
            mean_val = float(np.mean(result))
            assert mean_val > 5, f"seed={seed}: nearly all-black (mean={mean_val:.1f})"
            assert mean_val < 250, f"seed={seed}: nearly all-white (mean={mean_val:.1f})"


class TestMultiAtlasReferences:
    """Test multi-atlas reference training support."""

    def test_worker_accepts_multiple_references(self, config: DataGenerationConfig) -> None:
        """_worker_generate_batch should handle multiple memmap paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two small synthetic atlas volumes
            vol1 = np.random.default_rng(0).integers(0, 255, (100, 64, 64), dtype=np.uint8)
            vol2 = np.random.default_rng(1).integers(0, 255, (100, 64, 64), dtype=np.uint8)

            # Write to temp memmaps
            mm_path1 = Path(tmpdir) / "ref1.dat"
            mm_path2 = Path(tmpdir) / "ref2.dat"

            mm1 = np.memmap(str(mm_path1), dtype='uint8', mode='w+', shape=vol1.shape)
            mm1[:] = vol1[:]
            mm1.flush()
            del mm1

            mm2 = np.memmap(str(mm_path2), dtype='uint8', mode='w+', shape=vol2.shape)
            mm2[:] = vol2[:]
            mm2.flush()
            del mm2

            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            # Call worker with multiple references
            results = _worker_generate_batch(
                memmap_path=[str(mm_path1), str(mm_path2)],
                atlas_shape=vol1.shape,
                atlas_dtype=['uint8', 'uint8'],
                seeds=[42, 43],
                config_dict=config.model_dump(),
                output_dir=str(output_dir),
                ap_range=(0.0, float(vol1.shape[0])),
                reference_names=["default", "nissl"],
            )

            assert len(results) == 2
            for stem, metadata in results:
                assert "reference" in metadata
                assert metadata["reference"] in ["default", "nissl"]
                assert (output_dir / f"{stem}.png").exists()

    def test_metadata_contains_reference_field(self, config: DataGenerationConfig) -> None:
        """Generated samples should include 'reference' field in metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            vol = np.random.default_rng(0).integers(0, 255, (100, 64, 64), dtype=np.uint8)
            mm_path = Path(tmpdir) / "ref.dat"

            mm = np.memmap(str(mm_path), dtype='uint8', mode='w+', shape=vol.shape)
            mm[:] = vol[:]
            mm.flush()
            del mm

            output_dir = Path(tmpdir) / "output"
            output_dir.mkdir()

            results = _worker_generate_batch(
                memmap_path=str(mm_path),
                atlas_shape=vol.shape,
                atlas_dtype='uint8',
                seeds=[42],
                config_dict=config.model_dump(),
                output_dir=str(output_dir),
                ap_range=(0.0, float(vol.shape[0])),
                reference_names=["nissl"],
            )

            assert len(results) == 1
            _, metadata = results[0]
            assert metadata["reference"] == "nissl"
