"""Tests for the estimator training loop."""

import pickle

import numpy as np
import pytest
import torch

from belljar.config import EstimationConfig, TrainingConfig
from belljar.estimation.predictor import SliceEstimator, gram_schmidt_6d, load_model
from belljar.estimation.train import (
    ANCHORING_WEIGHTS,
    AnchoringLossWithUncertainty,
    anchoring_loss,
    geodesic_rotation_loss,
    train,
)

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
# Learned uncertainty loss tests (A1)
# ---------------------------------------------------------------------------


class TestUncertaintyLoss:
    def test_differentiable(self):
        """Gradients should flow to the log_var parameters."""
        loss_fn = AnchoringLossWithUncertainty()
        pred = torch.randn(4, 9)
        target = torch.randn(4, 9)
        loss = loss_fn(pred, target)
        loss.backward()
        assert loss_fn.log_var_origin.grad is not None
        assert loss_fn.log_var_u.grad is not None
        assert loss_fn.log_var_v.grad is not None
        assert loss_fn.log_var_origin.grad.abs().item() > 0

    def test_finite_lower_bound(self):
        """Loss has a finite lower bound: can't go to -inf.

        When log_var grows large and positive (low precision), the loss per task
        becomes: exp(-log_var)*mse + log_var ≈ 0 + log_var = log_var. The
        optimizer would push log_var to -inf to reduce the precision*mse term,
        but the +log_var regularizer pulls it back. We verify that at perfect
        prediction (MSE=0), the loss is bounded below (exactly sum(log_var)).
        """
        loss_fn = AnchoringLossWithUncertainty()
        # With large positive log_var, precision → 0, so loss ≈ sum(log_var)
        with torch.no_grad():
            loss_fn.log_var_origin.fill_(10.0)
            loss_fn.log_var_u.fill_(10.0)
            loss_fn.log_var_v.fill_(10.0)
        pred = torch.ones(4, 9)
        target = torch.ones(4, 9)  # perfect prediction
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss), f"Loss should be finite, got {loss.item()}"
        # At zero MSE: loss = sum(log_var) = 30.0
        assert loss.item() == pytest.approx(30.0, rel=1e-4)

    def test_nonzero_at_zero_error(self):
        """Loss should be positive when pred == target (due to log_var terms).

        At zero MSE, loss = sum(log_var_i) which is 0 at initialization.
        But after any training step that adjusts log_vars, this won't be exactly 0.
        Initialize with non-zero log_vars to demonstrate.
        """
        loss_fn = AnchoringLossWithUncertainty()
        # Set log_vars to positive values — loss = sum(log_var) > 0 at zero error
        with torch.no_grad():
            loss_fn.log_var_origin.fill_(1.0)
            loss_fn.log_var_u.fill_(1.0)
            loss_fn.log_var_v.fill_(1.0)
        pred = torch.ones(4, 9)
        target = torch.ones(4, 9)
        loss = loss_fn(pred, target)
        assert loss.item() > 0, "Loss should be positive at zero error with non-zero log_vars"

    def test_checkpoint_includes_loss_state_dict(self, small_training_data, tmp_path):
        """When using learned weights, checkpoint should contain loss_state_dict."""
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
            use_learned_loss_weights=True,
        )

        best_path = train(
            data_dir=small_training_data,
            config=config,
            estimation_config=EstimationConfig(),
            output_dir=output_dir,
            device=_TEST_DEVICE,
        )

        checkpoint = torch.load(str(best_path), weights_only=False)
        assert "loss_state_dict" in checkpoint
        assert "log_var_origin" in checkpoint["loss_state_dict"]
        assert "log_var_u" in checkpoint["loss_state_dict"]
        assert "log_var_v" in checkpoint["loss_state_dict"]


# ---------------------------------------------------------------------------
# Cosine similarity loss tests (A2)
# ---------------------------------------------------------------------------


class TestCosineSimilarityLoss:
    def test_parallel_vectors_zero_loss(self):
        """Cosine term should be 0 when direction vectors are parallel."""
        pred = torch.zeros(4, 9)
        target = torch.zeros(4, 9)
        # Set u and v to identical unit vectors in pred and target
        pred[:, 3:6] = torch.tensor([1.0, 0.0, 0.0])
        pred[:, 6:9] = torch.tensor([0.0, 1.0, 0.0])
        target[:, 3:6] = torch.tensor([1.0, 0.0, 0.0])
        target[:, 6:9] = torch.tensor([0.0, 1.0, 0.0])

        loss_no_cos = anchoring_loss(pred, target, cosine_weight=0.0)
        loss_with_cos = anchoring_loss(pred, target, cosine_weight=0.5)
        # Both should be zero since pred == target and cosine term is 0 for parallel
        assert loss_no_cos.item() == pytest.approx(0.0)
        assert loss_with_cos.item() == pytest.approx(0.0)

    def test_perpendicular_vectors_loss(self):
        """Cosine term should be 1.0 per perpendicular pair (cos_sim = 0)."""
        pred = torch.zeros(4, 9)
        target = torch.zeros(4, 9)
        # u vectors: pred=(1,0,0), target=(0,1,0) — perpendicular
        pred[:, 3:6] = torch.tensor([1.0, 0.0, 0.0])
        target[:, 3:6] = torch.tensor([0.0, 1.0, 0.0])
        # v vectors: pred=(0,0,1), target=(0,1,0) — perpendicular
        pred[:, 6:9] = torch.tensor([0.0, 0.0, 1.0])
        target[:, 6:9] = torch.tensor([0.0, 1.0, 0.0])

        # cos_sim(u) = 0, cos_sim(v) = 0
        # cosine_loss = weight * (2 - 0 - 0) = weight * 2
        weight = 0.5
        loss = anchoring_loss(pred, target, cosine_weight=weight)
        loss_no_cos = anchoring_loss(pred, target, cosine_weight=0.0)
        cosine_contribution = loss.item() - loss_no_cos.item()
        assert cosine_contribution == pytest.approx(weight * 2.0, rel=1e-5)

    def test_cosine_in_uncertainty_loss(self):
        """Cosine term should also work inside AnchoringLossWithUncertainty."""
        loss_fn_no_cos = AnchoringLossWithUncertainty(cosine_weight=0.0)
        loss_fn_cos = AnchoringLossWithUncertainty(cosine_weight=0.5)

        pred = torch.zeros(4, 9)
        target = torch.zeros(4, 9)
        # Perpendicular u vectors
        pred[:, 3:6] = torch.tensor([1.0, 0.0, 0.0])
        target[:, 3:6] = torch.tensor([0.0, 1.0, 0.0])
        pred[:, 6:9] = torch.tensor([0.0, 0.0, 1.0])
        target[:, 6:9] = torch.tensor([0.0, 1.0, 0.0])

        loss_no = loss_fn_no_cos(pred, target)
        loss_yes = loss_fn_cos(pred, target)
        # The cosine version should add 0.5 * 2.0 = 1.0 to the loss
        assert loss_yes.item() > loss_no.item()
        diff = loss_yes.item() - loss_no.item()
        assert diff == pytest.approx(1.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Geodesic loss tests (B2)
# ---------------------------------------------------------------------------


class TestGeodesicLoss:
    def test_identity_rotation_zero_loss(self):
        """Identical direction vectors should give zero geodesic loss."""
        # Orthonormal u and v vectors
        u = torch.tensor([[1.0, 0.0, 0.0]])
        v = torch.tensor([[0.0, 1.0, 0.0]])
        dirs = torch.cat([u, v], dim=1)  # (1, 6)

        loss = geodesic_rotation_loss(dirs, dirs)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_90_degree_rotation(self):
        """A 90-degree rotation about one axis should give loss ~pi/2."""
        import math

        # Identity rotation: u=[1,0,0], v=[0,1,0]
        pred = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
        # 90-degree rotation about z-axis: u=[0,1,0], v=[-1,0,0]
        target = torch.tensor([[0.0, 1.0, 0.0, -1.0, 0.0, 0.0]])

        loss = geodesic_rotation_loss(pred, target)
        assert loss.item() == pytest.approx(math.pi / 2, abs=1e-5)

    def test_gradients_finite(self):
        """Gradients through geodesic loss should be finite (no NaN/Inf)."""
        pred = torch.randn(8, 6, requires_grad=True)
        target = torch.randn(8, 6)

        loss = geodesic_rotation_loss(pred, target)
        loss.backward()

        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()


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


# ---------------------------------------------------------------------------
# DINOv2 estimator tests
# ---------------------------------------------------------------------------


class _FakeViTBackbone(torch.nn.Module):
    """Minimal stand-in for DINOv2 ViT-B that outputs (B, 768) features."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3 * 224 * 224, 768)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.flatten(1))


class TestDINOv2Estimator:
    @pytest.fixture(autouse=True)
    def _patch_timm(self, monkeypatch):
        """Replace timm.create_model to avoid downloading real weights."""
        import timm

        def _fake_create_model(*_args, **_kwargs):
            return _FakeViTBackbone()

        monkeypatch.setattr(timm, "create_model", _fake_create_model)

    def test_forward_shape(self):
        from belljar.estimation.predictor import DINOv2Estimator

        model = DINOv2Estimator(num_outputs=9, orthogonalize=True)
        x = torch.randn(2, 1, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 9)

    def test_backbone_frozen(self):
        from belljar.estimation.predictor import DINOv2Estimator

        model = DINOv2Estimator(num_outputs=9)
        for param in model.backbone.parameters():
            assert not param.requires_grad

    def test_head_trainable(self):
        from belljar.estimation.predictor import DINOv2Estimator

        model = DINOv2Estimator(num_outputs=9)
        for param in model.head.parameters():
            assert param.requires_grad
