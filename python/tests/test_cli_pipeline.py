"""Tests for the CLI, EstimateStep, and model_manager."""

import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from belljar.config import BelljarConfig, EstimationConfig
from belljar.estimation.model_manager import DEFAULT_MODEL_DIR, ensure_model
from belljar.pipeline.estimate import EstimateStep
from belljar.types import StepResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _noop_progress(current: int, total: int, message: str) -> None:
    pass


@pytest.fixture
def fake_model(tmp_path):
    """Create a minimal model checkpoint for testing."""
    from belljar.estimation.predictor import SliceEstimator

    model = SliceEstimator(num_outputs=9, dropout_rate=0.2, orthogonalize=True)
    model_path = tmp_path / "test_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 1,
            "val_loss": 0.5,
            "val_metrics": {"oz_mae": 0.1, "u_mae": 0.2, "v_mae": 0.15},
            "config": EstimationConfig().model_dump(),
            "training_config": {},
        },
        str(model_path),
    )
    return model_path


@pytest.fixture
def section_images(tmp_path):
    """Create a few fake section images."""
    import cv2

    img_dir = tmp_path / "sections"
    img_dir.mkdir()
    rng = np.random.default_rng(42)
    for i in range(5):
        img = rng.integers(0, 256, (256, 256), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"section_{i:03d}.png"), img)
    return img_dir


# ---------------------------------------------------------------------------
# Model manager tests
# ---------------------------------------------------------------------------


class TestModelManager:
    def test_explicit_path_found(self, fake_model):
        result = ensure_model(model_path=fake_model)
        assert result == fake_model

    def test_explicit_path_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="explicit path"):
            ensure_model(model_path=tmp_path / "nonexistent.pt")

    def test_no_model_available(self):
        with pytest.raises(FileNotFoundError, match="No model found"):
            ensure_model(model_path=None, gcs_uri=None)

    def test_cached_model(self, fake_model, tmp_path, monkeypatch):
        """Ensure model is found in the cache directory."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cached = cache_dir / "best_model.pt"
        # Copy the fake model to the cache location
        import shutil

        shutil.copy(fake_model, cached)

        monkeypatch.setattr(
            "belljar.estimation.model_manager.DEFAULT_MODEL_DIR", cache_dir
        )
        result = ensure_model(model_path=None)
        assert result == cached

    def test_gcs_download_gsutil_not_found(self):
        with pytest.raises(FileNotFoundError, match="gsutil not found"):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                ensure_model(gcs_uri="gs://fake-bucket/model.pt")


# ---------------------------------------------------------------------------
# EstimateStep tests
# ---------------------------------------------------------------------------


class TestEstimateStep:
    def test_name(self):
        config = BelljarConfig()
        step = EstimateStep(config)
        assert step.name == "Estimate"

    def test_validate_missing_input_dir(self):
        config = BelljarConfig()
        step = EstimateStep(config)
        errors = step.validate_inputs(output_dir="/tmp/out")
        assert any("input_dir" in e for e in errors)

    def test_validate_missing_output_dir(self, tmp_path):
        config = BelljarConfig()
        step = EstimateStep(config)
        errors = step.validate_inputs(input_dir=str(tmp_path))
        assert any("output_dir" in e for e in errors)

    def test_validate_nonexistent_input_dir(self, tmp_path):
        config = BelljarConfig()
        step = EstimateStep(config)
        errors = step.validate_inputs(
            input_dir=str(tmp_path / "nope"), output_dir=str(tmp_path / "out")
        )
        assert any("does not exist" in e for e in errors)

    def test_validate_nonexistent_model(self, tmp_path):
        config = BelljarConfig()
        step = EstimateStep(config)
        errors = step.validate_inputs(
            input_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            model_path=str(tmp_path / "no_model.pt"),
        )
        assert any("Model file not found" in e for e in errors)

    def test_validate_success(self, tmp_path, fake_model):
        config = BelljarConfig()
        step = EstimateStep(config)
        errors = step.validate_inputs(
            input_dir=str(tmp_path),
            output_dir=str(tmp_path / "out"),
            model_path=str(fake_model),
        )
        assert errors == []

    def test_run_no_images(self, tmp_path, fake_model):
        config = BelljarConfig()
        step = EstimateStep(config)
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        result = step.run(
            _noop_progress,
            input_dir=str(empty_dir),
            output_dir=str(tmp_path / "out"),
            model_path=str(fake_model),
        )
        assert not result.success
        assert any("No image files" in e for e in result.errors)

    def test_run_produces_estimates(self, section_images, fake_model, tmp_path):
        config = BelljarConfig()
        step = EstimateStep(config)
        output_dir = tmp_path / "estimates_out"

        result = step.run(
            _noop_progress,
            input_dir=str(section_images),
            output_dir=str(output_dir),
            model_path=str(fake_model),
        )

        assert result.success
        assert result.output_path is not None
        assert Path(result.output_path).exists()

        # Check output JSON structure
        with open(result.output_path) as f:
            data = json.load(f)

        assert "alignments" in data
        assert "metadata" in data
        assert len(data["alignments"]) == 5
        assert data["metadata"]["total_sections"] == 5

        # Check alignment fields
        alignment = data["alignments"][0]
        assert "section_name" in alignment
        assert "ap_position" in alignment
        assert "x_angle" in alignment
        assert "y_angle" in alignment

    def test_run_metrics(self, section_images, fake_model, tmp_path):
        config = BelljarConfig()
        step = EstimateStep(config)
        output_dir = tmp_path / "est_out"

        result = step.run(
            _noop_progress,
            input_dir=str(section_images),
            output_dir=str(output_dir),
            model_path=str(fake_model),
        )

        assert result.metrics["sections_estimated"] == 5
        assert result.metrics["sections_total"] == 5
        assert result.metrics["sections_failed"] == 0

    def test_output_usable_by_align(self, section_images, fake_model, tmp_path):
        """Estimates output should be parseable as AlignStep input."""
        from belljar.types import SliceAlignment

        config = BelljarConfig()
        step = EstimateStep(config)
        output_dir = tmp_path / "est_out"

        result = step.run(
            _noop_progress,
            input_dir=str(section_images),
            output_dir=str(output_dir),
            model_path=str(fake_model),
        )

        with open(result.output_path) as f:
            data = json.load(f)

        # Each alignment should be constructable as SliceAlignment
        for alignment_data in data["alignments"]:
            # Remove anchoring_vectors since it's not in SliceAlignment
            a = {k: v for k, v in alignment_data.items() if k != "anchoring_vectors"}
            sa = SliceAlignment(**a)
            assert isinstance(sa.ap_position, float)
            assert isinstance(sa.x_angle, float)


# ---------------------------------------------------------------------------
# Postprocessing tests (E1, E2, E3)
# ---------------------------------------------------------------------------


class TestConfidenceWeightedIntegration:
    """Tests for confidence-weighted angle integration (E1)."""

    def _make_predictions(self, x_angles, y_angles):
        return [
            {"x_angle": x, "y_angle": y, "z_position": float(i * 5)}
            for i, (x, y) in enumerate(zip(x_angles, y_angles))
        ]

    def test_high_uncertainty_adjusted_more(self):
        """Sections with high uncertainty should be pulled toward neighbors."""
        from belljar.estimation.postprocess import integrate_angles

        # 10 sections: all near 2.0 degrees except index 5 which is an outlier
        x_angles = [2.0] * 10
        y_angles = [2.0] * 10
        x_angles[5] = 10.0  # outlier
        y_angles[5] = 10.0

        preds = self._make_predictions(x_angles, y_angles)

        # High uncertainty on the outlier, low on others
        uncertainties = [0.1] * 10
        uncertainties[5] = 10.0  # very uncertain

        result = integrate_angles(preds, window_size=5, uncertainties=uncertainties)

        # The outlier's angle should be pulled substantially toward 2.0
        assert result[5]["x_angle"] < 5.0, (
            f"Expected outlier x_angle < 5.0, got {result[5]['x_angle']}"
        )
        assert result[5]["y_angle"] < 5.0, (
            f"Expected outlier y_angle < 5.0, got {result[5]['y_angle']}"
        )

    def test_no_uncertainty_falls_back(self):
        """When uncertainties=None, result should match rolling median."""
        from belljar.estimation.postprocess import integrate_angles

        x_angles = [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0]
        y_angles = [0.5, 1.0, 1.5, 1.0, 0.5, 1.0, 1.5, 1.0, 0.5, 1.0]
        preds = self._make_predictions(x_angles, y_angles)

        result_none = integrate_angles(preds, window_size=5, uncertainties=None)
        result_default = integrate_angles(preds, window_size=5)

        for a, b in zip(result_none, result_default):
            assert a["x_angle"] == pytest.approx(b["x_angle"])
            assert a["y_angle"] == pytest.approx(b["y_angle"])


class TestRANSACSpacing:
    """Tests for RANSAC spacing regularization (E2)."""

    def test_outlier_corrected(self):
        """RANSAC should correct a single large outlier better than linear blend."""
        from belljar.estimation.postprocess import regularize_spacing

        # 20 sections with expected 5-slice spacing, one massive outlier
        n = 20
        preds = []
        for i in range(n):
            z = 100.0 + i * 5.0
            preds.append({"z_position": z, "section_name": f"s{i:03d}"})

        # Inject outlier at index 10 (off by 100 slices)
        preds[10]["z_position"] = 100.0 + 10 * 5.0 + 100.0

        # RANSAC should correct the outlier substantially
        result_ransac = regularize_spacing(preds, method="ransac")
        expected_z = 100.0 + 10 * 5.0  # what it should be

        ransac_error = abs(result_ransac[10]["z_position"] - expected_z)
        assert ransac_error < 10.0, (
            f"RANSAC error {ransac_error} should be < 10 for a 100-slice outlier"
        )

        # Linear blend only halves the error (alpha=0.5 blend)
        result_linear = regularize_spacing(preds, method="linear")
        linear_error = abs(result_linear[10]["z_position"] - expected_z)
        assert ransac_error < linear_error, (
            "RANSAC should correct outlier better than linear blend"
        )

    def test_preserves_raw_position(self):
        """Regularized output should preserve z_position_raw."""
        from belljar.estimation.postprocess import regularize_spacing

        preds = [
            {"z_position": float(i * 5), "section_name": f"s{i}"}
            for i in range(10)
        ]
        result = regularize_spacing(preds, method="ransac")
        for i, r in enumerate(result):
            assert r["z_position_raw"] == float(i * 5)


class TestOrthogonalityEnforcement:
    """Tests for orthogonality enforcement via numpy Gram-Schmidt (E3)."""

    def test_nonorthogonal_input_becomes_orthogonal(self):
        """u . v should be approximately 0 after enforcement."""
        from belljar.estimation.postprocess import enforce_orthogonality

        # u and v are NOT orthogonal (dot product != 0)
        preds = [
            {
                "anchoring_vectors": {
                    "ox": 0.5, "oy": 0.5, "oz": 0.3,
                    "ux": 1.0, "uy": 0.5, "uz": 0.0,
                    "vx": 0.5, "vy": 1.0, "vz": 0.2,
                },
                "z_position": 100.0,
            }
        ]

        result = enforce_orthogonality(preds)
        av = result[0]["anchoring_vectors"]
        u = np.array([av["ux"], av["uy"], av["uz"]])
        v = np.array([av["vx"], av["vy"], av["vz"]])

        dot = np.dot(u, v)
        assert abs(dot) < 1e-7, f"u . v = {dot}, expected ~0"

    def test_unit_length(self):
        """|u| and |v| should be approximately 1 after enforcement."""
        from belljar.estimation.postprocess import enforce_orthogonality

        preds = [
            {
                "anchoring_vectors": {
                    "ox": 0.5, "oy": 0.5, "oz": 0.3,
                    "ux": 3.0, "uy": 0.0, "uz": 0.0,
                    "vx": 0.0, "vy": 2.0, "vz": 1.0,
                },
                "z_position": 100.0,
            }
        ]

        result = enforce_orthogonality(preds)
        av = result[0]["anchoring_vectors"]
        u = np.array([av["ux"], av["uy"], av["uz"]])
        v = np.array([av["vx"], av["vy"], av["vz"]])

        assert abs(np.linalg.norm(u) - 1.0) < 1e-7, (
            f"|u| = {np.linalg.norm(u)}, expected 1.0"
        )
        assert abs(np.linalg.norm(v) - 1.0) < 1e-7, (
            f"|v| = {np.linalg.norm(v)}, expected 1.0"
        )

    def test_flat_anchoring_format(self):
        """Should also handle flat 9-element 'anchoring' list."""
        from belljar.estimation.postprocess import enforce_orthogonality

        preds = [
            {
                "anchoring": [0.5, 0.5, 0.3, 1.0, 0.5, 0.0, 0.5, 1.0, 0.2],
                "z_position": 100.0,
            }
        ]

        result = enforce_orthogonality(preds)
        a = result[0]["anchoring"]
        u = np.array(a[3:6])
        v = np.array(a[6:9])

        assert abs(np.dot(u, v)) < 1e-7
        assert abs(np.linalg.norm(u) - 1.0) < 1e-7
        assert abs(np.linalg.norm(v) - 1.0) < 1e-7

    def test_no_anchoring_passes_through(self):
        """Predictions without anchoring data should be unmodified."""
        from belljar.estimation.postprocess import enforce_orthogonality

        preds = [{"z_position": 100.0, "x_angle": 2.0}]
        result = enforce_orthogonality(preds)
        assert result[0] == preds[0]


# ---------------------------------------------------------------------------
# Server integration tests
# ---------------------------------------------------------------------------


class TestServerIntegration:
    def test_estimate_handler_registered(self):
        from belljar.server import BelljarServer

        server = BelljarServer()
        assert "pipeline.estimate" in server._handlers

    def test_estimate_step_in_step_classes(self):
        from belljar.server import BelljarServer

        server = BelljarServer()
        step = server._get_step("estimate")
        assert step.name == "Estimate"
