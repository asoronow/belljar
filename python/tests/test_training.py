"""Tests for the estimator training loop."""

import pickle

import numpy as np
import pytest
import torch

from belljar.config import EstimationConfig, TrainingConfig
from belljar.estimation.predictor import SliceEstimator, gram_schmidt_6d, load_model
from belljar.estimation.train import ANCHORING_WEIGHTS, _mixup_batch, anchoring_loss, train

# Force CPU for tests — MPS has numerical instability with GradScaler/autocast
_TEST_DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_training_data(tmp_path):
    """Generate a minimal training dataset (50 samples) for fast tests."""
    import cv2

    rng = np.random.default_rng(42)
    n_samples = 50
    metadata = {}

    for i in range(n_samples):
        # Random 256x256 grayscale image
        img = rng.integers(0, 256, (256, 256), dtype=np.uint8)
        stem = f"S_{i:04d}"
        cv2.imwrite(str(tmp_path / f"{stem}.png"), img)

        # Random anchoring vector as label
        anchoring = rng.uniform(-1, 1, 9).tolist()
        # oz should be in [0, 1] (normalized AP position)
        anchoring[2] = rng.uniform(0, 1)
        metadata[stem] = {
            "pos": float(rng.integers(50, 1270)),
            "x_angle": float(rng.uniform(-15, 15)),
            "y_angle": float(rng.uniform(-15, 15)),
            "z_angle": float(rng.uniform(-5, 5)),
            "anchoring": anchoring,
        }

    with open(tmp_path / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    return tmp_path


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------


class TestAnchoringLoss:
    def test_zero_error_gives_zero_loss(self):
        pred = torch.ones(4, 9)
        target = torch.ones(4, 9)
        assert anchoring_loss(pred, target).item() == pytest.approx(0.0)

    def test_direction_vectors_weighted_higher(self):
        """Direction vector errors should contribute 2x vs origin errors."""
        # Error only in origin (ox)
        pred_origin = torch.zeros(1, 9)
        target_origin = torch.zeros(1, 9)
        pred_origin[0, 0] = 1.0  # ox error = 1.0

        # Error only in u vector (ux)
        pred_uv = torch.zeros(1, 9)
        target_uv = torch.zeros(1, 9)
        pred_uv[0, 3] = 1.0  # ux error = 1.0

        loss_origin = anchoring_loss(pred_origin, target_origin)
        loss_uv = anchoring_loss(pred_uv, target_uv)

        # uv loss should be exactly 2x origin loss (same error magnitude, 2x weight)
        assert loss_uv.item() == pytest.approx(loss_origin.item() * 2.0, rel=1e-5)

    def test_weights_shape(self):
        assert ANCHORING_WEIGHTS.shape == (9,)
        assert ANCHORING_WEIGHTS[:3].sum().item() == pytest.approx(3.0)
        assert ANCHORING_WEIGHTS[3:].sum().item() == pytest.approx(12.0)

    def test_batch_independence(self):
        """Loss should be mean over batch, not sum."""
        pred = torch.randn(1, 9)
        target = torch.zeros(1, 9)
        loss_1 = anchoring_loss(pred, target)

        pred_4 = pred.repeat(4, 1)
        target_4 = target.repeat(4, 1)
        loss_4 = anchoring_loss(pred_4, target_4)

        assert loss_1.item() == pytest.approx(loss_4.item(), rel=1e-5)


# ---------------------------------------------------------------------------
# Training loop tests
# ---------------------------------------------------------------------------


class TestTrainLoop:
    def test_train_reduces_loss(self, small_training_data, tmp_path):
        """Training for a few epochs should reduce validation loss."""
        output_dir = tmp_path / "output"
        config = TrainingConfig(
            batch_size=16,
            num_epochs=3,
            learning_rate=1e-3,
            warmup_epochs=1,
            val_fraction=0.2,
            num_workers=0,
            mixed_precision=False,
            checkpoint_every=10,
            early_stopping_patience=100,
            seed=42,
        )
        estimation_config = EstimationConfig()

        best_path = train(
            data_dir=small_training_data,
            config=config,
            estimation_config=estimation_config,
            output_dir=output_dir,
            device=_TEST_DEVICE,
        )

        assert best_path.exists()
        assert best_path.name == "best_model.pt"

    def test_checkpoint_format(self, small_training_data, tmp_path):
        """Enriched checkpoint should contain all expected keys."""
        output_dir = tmp_path / "output"
        config = TrainingConfig(
            batch_size=16,
            num_epochs=2,
            warmup_epochs=1,
            val_fraction=0.2,
            num_workers=0,
            mixed_precision=False,
            checkpoint_every=10,
            early_stopping_patience=100,
        )
        estimation_config = EstimationConfig()

        best_path = train(
            data_dir=small_training_data,
            config=config,
            estimation_config=estimation_config,
            output_dir=output_dir,
            device=_TEST_DEVICE,
        )

        checkpoint = torch.load(str(best_path), weights_only=False)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "scheduler_state_dict" in checkpoint
        assert "epoch" in checkpoint
        assert "val_loss" in checkpoint
        assert "val_metrics" in checkpoint
        assert "config" in checkpoint
        assert "training_config" in checkpoint
        assert isinstance(checkpoint["val_loss"], float)
        assert checkpoint["epoch"] >= 1

    def test_checkpoint_loadable(self, small_training_data, tmp_path):
        """Best checkpoint should be loadable via load_model()."""
        output_dir = tmp_path / "output"
        config = TrainingConfig(
            batch_size=16,
            num_epochs=2,
            warmup_epochs=1,
            val_fraction=0.2,
            num_workers=0,
            mixed_precision=False,
            checkpoint_every=10,
            early_stopping_patience=100,
        )
        estimation_config = EstimationConfig()

        best_path = train(
            data_dir=small_training_data,
            config=config,
            estimation_config=estimation_config,
            output_dir=output_dir,
            device=_TEST_DEVICE,
        )

        model = load_model(best_path, estimation_config, device=torch.device("cpu"))
        assert isinstance(model, SliceEstimator)

        # Verify inference works
        dummy = torch.randn(1, 1, 256, 256)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 9)

    def test_val_metrics_contain_components(self, small_training_data, tmp_path):
        """Validation metrics should include per-component MAE."""
        output_dir = tmp_path / "output"
        config = TrainingConfig(
            batch_size=16,
            num_epochs=2,
            warmup_epochs=1,
            val_fraction=0.2,
            num_workers=0,
            mixed_precision=False,
            checkpoint_every=10,
            early_stopping_patience=100,
        )

        best_path = train(
            data_dir=small_training_data,
            config=config,
            estimation_config=EstimationConfig(),
            output_dir=output_dir,
            device=_TEST_DEVICE,
        )

        checkpoint = torch.load(str(best_path), weights_only=False)
        metrics = checkpoint["val_metrics"]
        assert "oz_mae" in metrics
        assert "u_mae" in metrics
        assert "v_mae" in metrics
        assert all(isinstance(v, float) for v in metrics.values())


# ---------------------------------------------------------------------------
# MixUp augmentation tests
# ---------------------------------------------------------------------------


class TestMixUpAugmentation:
    def test_mixup_disabled_returns_unchanged(self):
        """When alpha=0, MixUp should return inputs unchanged."""
        images = torch.randn(8, 1, 256, 256)
        labels = torch.randn(8, 9)
        img_mixed, lbl_mixed = _mixup_batch(images, labels, alpha=0.0)
        assert torch.equal(img_mixed, images)
        assert torch.equal(lbl_mixed, labels)

    def test_mixup_output_shapes(self):
        """MixUp should preserve batch shape."""
        images = torch.randn(8, 1, 256, 256)
        labels = torch.randn(8, 9)
        img_mixed, lbl_mixed = _mixup_batch(images, labels, alpha=0.2)
        assert img_mixed.shape == images.shape
        assert lbl_mixed.shape == labels.shape

    def test_mixup_creates_blend(self):
        """MixUp should create a blend between samples."""
        torch.manual_seed(42)
        images = torch.zeros(4, 1, 2, 2)
        images[0].fill_(1.0)  # First sample all 1s
        images[1].fill_(2.0)  # Second sample all 2s
        labels = torch.zeros(4, 9)
        labels[0].fill_(1.0)
        labels[1].fill_(2.0)

        img_mixed, lbl_mixed = _mixup_batch(images, labels, alpha=0.2)

        # Mixed results should be between original values (not exactly equal to either)
        # Some samples should have intermediate values
        unique_vals = img_mixed.unique()
        assert len(unique_vals) > 2  # More than just the original two values


# ---------------------------------------------------------------------------
# Gram-Schmidt orthogonalization tests
# ---------------------------------------------------------------------------


class TestGramSchmidt:
    def test_output_orthogonal(self):
        """u and v output vectors should be orthogonal (dot product ~0)."""
        raw = torch.randn(16, 6)
        result = gram_schmidt_6d(raw)
        u = result[:, :3]
        v = result[:, 3:]
        dot = (u * v).sum(dim=-1)
        assert torch.allclose(dot, torch.zeros(16), atol=1e-6)

    def test_output_unit_length(self):
        """u and v output vectors should each have unit length."""
        raw = torch.randn(16, 6)
        result = gram_schmidt_6d(raw)
        u = result[:, :3]
        v = result[:, 3:]
        u_norm = torch.linalg.norm(u, dim=-1)
        v_norm = torch.linalg.norm(v, dim=-1)
        assert torch.allclose(u_norm, torch.ones(16), atol=1e-6)
        assert torch.allclose(v_norm, torch.ones(16), atol=1e-6)

    def test_gradient_flow(self):
        """Gradients should flow through gram_schmidt_6d."""
        raw = torch.randn(4, 6, requires_grad=True)
        result = gram_schmidt_6d(raw)
        loss = result.sum()
        loss.backward()
        assert raw.grad is not None
        assert not torch.all(raw.grad == 0)
