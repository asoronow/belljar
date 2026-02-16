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
