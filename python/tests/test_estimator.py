"""Tests for slice position estimator modules."""

import numpy as np
import pytest
import torch

from belljar.config import EstimationConfig
from belljar.estimation.predictor import (
    SliceEstimator,
    anchoring_to_legacy,
    legacy_to_anchoring,
    _preprocess_image,
)
from belljar.estimation.postprocess import (
    integrate_angles,
    regularize_spacing,
    denormalize_predictions,
)


class TestSliceEstimator:
    """Tests for the SliceEstimator model."""

    def test_output_shape(self) -> None:
        model = SliceEstimator(num_outputs=9)
        x = torch.randn(2, 1, 256, 256)
        out = model(x)
        assert out.shape == (2, 9)

    def test_legacy_output_shape(self) -> None:
        model = SliceEstimator(num_outputs=3)
        x = torch.randn(1, 1, 256, 256)
        out = model(x)
        assert out.shape == (1, 3)

    def test_feature_extraction(self) -> None:
        model = SliceEstimator(num_outputs=9)
        x = torch.randn(1, 1, 256, 256)
        features = model.extract_features(x)
        assert features.shape == (1, 2048)

    def test_dropout_produces_variation(self) -> None:
        """MC Dropout should produce different outputs in train mode."""
        model = SliceEstimator(num_outputs=9, dropout_rate=0.5)
        x = torch.randn(1, 1, 256, 256)
        model.train()
        with torch.no_grad():
            outputs = [model(x).numpy().copy() for _ in range(5)]
        # With 50% dropout, outputs should vary
        stacked = np.array(outputs)
        assert stacked.std(axis=0).sum() > 0

    def test_eval_mode_is_deterministic(self) -> None:
        model = SliceEstimator(num_outputs=9)
        x = torch.randn(1, 1, 256, 256)
        model.eval()
        with torch.no_grad():
            o1 = model(x).numpy()
            o2 = model(x).numpy()
        np.testing.assert_array_equal(o1, o2)


class TestPreprocessImage:
    def test_output_shape(self) -> None:
        config = EstimationConfig()
        img = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        tensor = _preprocess_image(img, config)
        assert tensor.shape == (1, 1, 256, 256)

    def test_handles_bgr_input(self) -> None:
        config = EstimationConfig()
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        tensor = _preprocess_image(img, config)
        assert tensor.shape == (1, 1, 256, 256)

    def test_handles_uint16_input(self) -> None:
        config = EstimationConfig()
        img = np.random.randint(0, 65535, (512, 512), dtype=np.uint16)
        tensor = _preprocess_image(img, config)
        assert tensor.shape == (1, 1, 256, 256)


class TestAnchoringConversion:
    def test_roundtrip(self) -> None:
        """Converting to anchoring and back should approximately recover the originals."""
        z_pos, x_angle, y_angle = 500.0, 3.0, -2.0
        anchoring = legacy_to_anchoring(z_pos, x_angle, y_angle)
        assert len(anchoring) == 9
        z_rec, x_rec, y_rec = anchoring_to_legacy(anchoring)
        assert abs(z_rec - z_pos) < 1.0

    def test_zero_angles(self) -> None:
        anchoring = legacy_to_anchoring(662.0, 0.0, 0.0)
        z_rec, x_rec, y_rec = anchoring_to_legacy(anchoring)
        assert abs(z_rec - 662.0) < 1.0
        assert abs(x_rec) < 0.1
        assert abs(y_rec) < 0.1


class TestAngleIntegration:
    def test_smooths_outliers(self) -> None:
        preds = [
            {"z_position": 100, "x_angle": 2.0, "y_angle": 1.0},
            {"z_position": 105, "x_angle": 2.1, "y_angle": 1.1},
            {"z_position": 110, "x_angle": 15.0, "y_angle": 1.0},  # outlier
            {"z_position": 115, "x_angle": 2.0, "y_angle": 0.9},
            {"z_position": 120, "x_angle": 2.1, "y_angle": 1.1},
        ]
        smoothed = integrate_angles(preds, window_size=3)
        # The outlier at index 2 should be smoothed toward 2.0
        assert abs(smoothed[2]["x_angle"] - 2.0) < abs(15.0 - 2.0)

    def test_preserves_raw_values(self) -> None:
        preds = [
            {"z_position": 100, "x_angle": 2.0, "y_angle": 1.0},
            {"z_position": 105, "x_angle": 3.0, "y_angle": 2.0},
            {"z_position": 110, "x_angle": 4.0, "y_angle": 3.0},
        ]
        smoothed = integrate_angles(preds, window_size=3)
        for s in smoothed:
            assert "x_angle_raw" in s
            assert "y_angle_raw" in s

    def test_short_list_unchanged(self) -> None:
        preds = [{"z_position": 100, "x_angle": 2.0, "y_angle": 1.0}]
        result = integrate_angles(preds)
        assert result == preds


class TestRegularizeSpacing:
    def test_smooths_positions(self) -> None:
        # Positions with a bump that should get smoothed
        preds = [
            {"z_position": float(i * 5)} for i in range(10)
        ]
        preds[5]["z_position"] = 50.0  # jump from expected 25
        result = regularize_spacing(preds)
        # The jump at index 5 should be partially smoothed
        assert result[5]["z_position"] < 50.0

    def test_preserves_raw(self) -> None:
        preds = [{"z_position": float(i)} for i in range(5)]
        result = regularize_spacing(preds)
        for r in result:
            assert "z_position_raw" in r


class TestDenormalize:
    def test_denormalize_position(self) -> None:
        preds = [{"z_position": 0.5, "x_angle": 0.5, "y_angle": 0.5}]
        result = denormalize_predictions(preds)
        assert result[0]["z_position"] == 662.0
        assert result[0]["x_angle"] == 0.0
        assert result[0]["y_angle"] == 0.0

    def test_denormalize_edges(self) -> None:
        preds = [{"z_position": 0.0, "x_angle": 0.0, "y_angle": 1.0}]
        result = denormalize_predictions(preds)
        assert result[0]["z_position"] == 0.0
        assert result[0]["x_angle"] == -10.0
        assert result[0]["y_angle"] == 10.0
