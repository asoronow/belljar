"""Training loop for the slice position estimator.

Trains the ResNet50-based SliceEstimator on synthetic atlas slices with
9-value anchoring vector labels. Supports AMP mixed precision, cosine LR
with warmup, wandb logging, checkpointing, and early stopping.

Usage (programmatic):
    from belljar.estimation.train import train
    from belljar.config import TrainingConfig, EstimationConfig
    best_path = train(data_dir, TrainingConfig(), EstimationConfig(), output_dir)

Usage (CLI):
    python scripts/train_estimator.py --data-dir /data/training --output-dir /data/model
"""

from __future__ import annotations

import logging
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms

from belljar.config import EstimationConfig, TrainingConfig
from belljar.estimation.dataset import AngledAtlasDataset
from belljar.estimation.predictor import SliceEstimator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

# Weights for the 9 anchoring components:
# Origin (ox, oy, oz) gets weight 1.0 -- position in atlas space.
# Direction vectors u and v get weight 2.0 -- small errors in plane
# orientation propagate into large spatial errors at image edges
# (256px half-width * 0.01 error ~ 2.56px displacement).
ANCHORING_WEIGHTS = torch.tensor(
    [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
    dtype=torch.float32,
)


def anchoring_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    cosine_weight: float = 0.0,
) -> torch.Tensor:
    """Weighted MSE loss for 9-value anchoring vectors.

    Args:
        pred: Predicted anchoring vectors, shape (B, 9).
        target: Ground truth anchoring vectors, shape (B, 9).
        cosine_weight: Weight for cosine similarity loss on direction vectors.
            When > 0, adds ``w * (2 - cos_sim(u) - cos_sim(v))`` to the loss.

    Returns:
        Scalar loss tensor.
    """
    weights = ANCHORING_WEIGHTS.to(pred.device)
    mse = (weights * (pred - target) ** 2).mean()
    if cosine_weight > 0.0:
        cos_u = F.cosine_similarity(pred[:, 3:6], target[:, 3:6], dim=1).mean()
        cos_v = F.cosine_similarity(pred[:, 6:9], target[:, 6:9], dim=1).mean()
        mse = mse + cosine_weight * (2.0 - cos_u - cos_v)
    return mse


class AnchoringLossWithUncertainty(nn.Module):
    """Learned multi-task uncertainty weights (Kendall & Gal, CVPR 2018).

    Instead of fixed weights for origin / u-vector / v-vector, learns three
    log-variance parameters that automatically balance the loss contributions.
    Loss per task: ``precision * mse + log_var`` where ``precision = exp(-log_var)``.
    The ``log_var`` term acts as a regularizer that prevents the precision from
    collapsing to zero (which would make the loss go to -inf).
    """

    def __init__(self, cosine_weight: float = 0.0) -> None:
        super().__init__()
        self.log_var_origin = nn.Parameter(torch.zeros(1))
        self.log_var_u = nn.Parameter(torch.zeros(1))
        self.log_var_v = nn.Parameter(torch.zeros(1))
        self.cosine_weight = cosine_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        slices = [
            (0, 3, self.log_var_origin),
            (3, 6, self.log_var_u),
            (6, 9, self.log_var_v),
        ]
        loss = torch.tensor(0.0, device=pred.device)
        for s, e, log_var in slices:
            mse = ((pred[:, s:e] - target[:, s:e]) ** 2).mean()
            precision = torch.exp(-log_var)
            loss = loss + precision * mse + log_var

        if self.cosine_weight > 0.0:
            cos_u = F.cosine_similarity(pred[:, 3:6], target[:, 3:6], dim=1).mean()
            cos_v = F.cosine_similarity(pred[:, 6:9], target[:, 6:9], dim=1).mean()
            loss = loss + self.cosine_weight * (2.0 - cos_u - cos_v)

        return loss

    def get_weights(self) -> dict[str, float]:
        """Return the effective precision weights (exp(-log_var)) for logging."""
        return {
            "weight_origin": torch.exp(-self.log_var_origin).item(),
            "weight_u": torch.exp(-self.log_var_u).item(),
            "weight_v": torch.exp(-self.log_var_v).item(),
        }


def geodesic_rotation_loss(
    pred_dirs: torch.Tensor, target_dirs: torch.Tensor
) -> torch.Tensor:
    """Geodesic distance on SO(3) between predicted and target rotation matrices.

    Constructs rotation matrices from direction vectors u and v by computing
    w = u x v (cross product) to form R = [u | v | w]. The geodesic distance
    is: arccos(clamp((trace(R_pred^T @ R_gt) - 1) / 2, -1, 1)).

    Args:
        pred_dirs: Predicted direction vectors, shape (B, 6) — [u(3), v(3)].
        target_dirs: Ground truth direction vectors, shape (B, 6).

    Returns:
        Scalar mean geodesic distance in radians.
    """
    pred_u = pred_dirs[:, :3]
    pred_v = pred_dirs[:, 3:]
    pred_w = torch.linalg.cross(pred_u, pred_v, dim=-1)

    target_u = target_dirs[:, :3]
    target_v = target_dirs[:, 3:]
    target_w = torch.linalg.cross(target_u, target_v, dim=-1)

    # Build rotation matrices: (B, 3, 3) with columns [u, v, w]
    R_pred = torch.stack([pred_u, pred_v, pred_w], dim=-1)
    R_target = torch.stack([target_u, target_v, target_w], dim=-1)

    # R_pred^T @ R_target
    R_rel = torch.bmm(R_pred.transpose(1, 2), R_target)

    # trace of each (B, 3, 3) matrix
    trace = R_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

    # Geodesic distance: arccos(clamp((trace - 1) / 2, -1, 1))
    cos_angle = torch.clamp((trace - 1.0) / 2.0, -1.0, 1.0)
    angle = torch.acos(cos_angle)

    return angle.mean()


def _compute_loss(
    preds: torch.Tensor,
    labels: torch.Tensor,
    loss_type: str,
    direction_cosine_weight: float,
    loss_fn: AnchoringLossWithUncertainty | None = None,
) -> torch.Tensor:
    """Compute loss based on loss_type config.

    Args:
        preds: Predicted anchoring vectors, shape (B, 9).
        labels: Ground truth anchoring vectors, shape (B, 9).
        loss_type: One of "mse", "cosine", "geodesic".
        direction_cosine_weight: Weight for cosine term (only used with "cosine").
        loss_fn: Optional learned uncertainty weights module.

    Returns:
        Scalar loss tensor.
    """
    if loss_type == "geodesic":
        # MSE for origin (0:3) + geodesic for directions (3:9)
        origin_weights = ANCHORING_WEIGHTS[:3].to(preds.device)
        origin_mse = (origin_weights * (preds[:, :3] - labels[:, :3]) ** 2).mean()
        geo_loss = geodesic_rotation_loss(preds[:, 3:9], labels[:, 3:9])
        return origin_mse + geo_loss

    # For "mse" and "cosine", use learned uncertainty weights if provided
    if loss_fn is not None:
        return loss_fn(preds, labels)

    # Fallback to fixed-weight losses
    if loss_type == "cosine":
        base = anchoring_loss(preds, labels, cosine_weight=direction_cosine_weight)
        return base

    # Default: "mse"
    return anchoring_loss(preds, labels, cosine_weight=0.0)


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------


def _mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply MixUp augmentation to a batch.

    Linearly interpolates random pairs of samples using a Beta-distributed
    mixing coefficient. When alpha <= 0, returns inputs unchanged.

    Args:
        images: Batch of images, shape (B, C, H, W).
        labels: Batch of labels, shape (B, D).
        alpha: Beta distribution parameter. Higher = more mixing.

    Returns:
        Tuple of (mixed_images, mixed_labels).
    """
    if alpha <= 0.0:
        return images, labels

    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    perm = torch.randperm(images.size(0), device=images.device)
    images_mix = lam * images + (1.0 - lam) * images[perm]
    labels_mix = lam * labels + (1.0 - lam) * labels[perm]
    return images_mix, labels_mix


def _compute_sample_weights(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    top_fraction: float,
    batch_size: int,
) -> list[float]:
    """Compute per-sample weights for hard negative mining.

    Runs inference on the full training set, records per-sample loss,
    then assigns weight 3.0 to the top ``top_fraction`` highest-loss
    samples and 1.0 to the rest.

    Args:
        model: Trained model (set to eval mode internally).
        dataset: Training dataset.
        device: Compute device.
        top_fraction: Fraction of hardest samples to upweight.
        batch_size: Batch size for inference.

    Returns:
        List of per-sample weights (length = len(dataset)).
    """
    model.eval()
    weights_tensor = ANCHORING_WEIGHTS.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    per_sample_losses: list[float] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            preds = model(images)
            # Per-sample weighted MSE (mean over 9 components, not batch)
            sample_losses = (weights_tensor * (preds - labels) ** 2).mean(dim=1)
            per_sample_losses.extend(sample_losses.cpu().tolist())

    model.train()

    # Threshold: top_fraction get weight 3.0, rest get 1.0
    n_hard = max(1, int(len(per_sample_losses) * top_fraction))
    threshold = sorted(per_sample_losses, reverse=True)[min(n_hard - 1, len(per_sample_losses) - 1)]
    sample_weights = [3.0 if loss >= threshold else 1.0 for loss in per_sample_losses]
    return sample_weights


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    data_dir: Path,
    config: TrainingConfig,
    estimation_config: EstimationConfig,
    output_dir: Path,
    *,
    device: torch.device | None = None,
) -> Path:
    """Train the slice position estimator.

    Args:
        data_dir: Directory with PNGs + metadata.pkl from data generation.
        config: Training hyperparameters.
        estimation_config: Model/preprocessing configuration.
        output_dir: Directory for checkpoints and logs.

    Returns:
        Path to the best checkpoint file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if device is None:
        device = _get_device()
    logger.info("Training on device: %s", device)

    # -- Dataset ---------------------------------------------------------------
    tx = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[estimation_config.normalization_mean],
            std=[estimation_config.normalization_std],
        ),
    ])

    full_dataset = AngledAtlasDataset(
        data_path=data_dir,
        transform=tx,
        output_format="anchoring",
        preprocessing=estimation_config.preprocessing,
    )

    n_val = max(1, int(len(full_dataset) * config.val_fraction))
    n_train = len(full_dataset) - n_val
    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(config.seed),
    )

    use_pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=config.num_workers > 0,
    )

    logger.info("Dataset: %d train, %d val", n_train, n_val)

    # -- Model -----------------------------------------------------------------
    model = SliceEstimator(
        num_outputs=9,
        dropout_rate=0.2,
        orthogonalize=estimation_config.orthogonalize_directions,
    ).to(device)

    # -- Loss function ---------------------------------------------------------
    loss_fn: AnchoringLossWithUncertainty | None = None
    if config.use_learned_loss_weights:
        loss_fn = AnchoringLossWithUncertainty(
            cosine_weight=config.direction_cosine_weight,
        ).to(device)

    # -- Optimizer + scheduler -------------------------------------------------
    params = list(model.parameters())
    if loss_fn is not None:
        params = params + list(loss_fn.parameters())
    optimizer = torch.optim.AdamW(
        params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    steps_per_epoch = len(train_loader)
    warmup_steps = config.warmup_epochs * steps_per_epoch
    total_steps = config.num_epochs * steps_per_epoch

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_steps,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=config.min_lr,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps],
    )

    # -- AMP scaler ------------------------------------------------------------
    use_amp = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.float16 if use_amp else torch.float32

    # -- wandb (optional) ------------------------------------------------------
    wandb_run = _init_wandb(config, estimation_config, n_train, n_val)

    # -- Training --------------------------------------------------------------
    best_val_loss = float("inf")
    best_epoch = -1
    patience_counter = 0
    best_path = output_dir / "best_model.pt"

    logger.info(
        "Starting training: %d epochs, batch_size=%d, lr=%.1e, AMP=%s, loss_type=%s",
        config.num_epochs, config.batch_size, config.learning_rate, use_amp, config.loss_type,
    )

    for epoch in range(config.num_epochs):
        t0 = time.time()

        # -- Train epoch -------------------------------------------------------
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        clip_count = 0
        step_count = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            images, labels = _mixup_batch(images, labels, config.mixup_alpha)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                preds = model(images)
                loss = _compute_loss(
                    preds, labels,
                    loss_type=config.loss_type,
                    direction_cosine_weight=config.direction_cosine_weight,
                    loss_fn=loss_fn,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss_sum += loss.item() * images.size(0)
            train_count += images.size(0)

            # Gradient clip diagnostics
            norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            if norm_val > 10.0:
                clip_count += 1
            step_count += 1

        train_loss = train_loss_sum / max(train_count, 1)
        clip_fraction = clip_count / max(step_count, 1)

        # -- Validation epoch --------------------------------------------------
        val_loss, val_metrics = _validate(
            model, val_loader, device, use_amp, amp_dtype,
            loss_type=config.loss_type,
            direction_cosine_weight=config.direction_cosine_weight,
            loss_fn=loss_fn,
        )

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %d/%d: train_loss=%.6f val_loss=%.6f lr=%.2e (%.1fs)",
            epoch + 1, config.num_epochs, train_loss, val_loss, lr, elapsed,
        )

        # -- wandb logging -----------------------------------------------------
        if wandb_run is not None:
            import wandb

            log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "train/grad_clip_fraction": clip_fraction,
                "val/loss": val_loss,
                "val/oz_mae": val_metrics["oz_mae"],
                "val/u_mae": val_metrics["u_mae"],
                "val/v_mae": val_metrics["v_mae"],
                "lr": lr,
                "grad_norm": norm_val,
                "epoch_time_s": elapsed,
            }
            if loss_fn is not None:
                w = loss_fn.get_weights()
                log_dict["val/weight_origin"] = w["weight_origin"]
                log_dict["val/weight_u"] = w["weight_u"]
                log_dict["val/weight_v"] = w["weight_v"]
            wandb.log(log_dict)

        # -- Checkpointing -----------------------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            _save_checkpoint(
                best_path, model, optimizer, scheduler, epoch + 1,
                val_loss, val_metrics, estimation_config, config,
                loss_fn=loss_fn,
            )
            logger.info("New best model (epoch %d, val_loss=%.6f)", best_epoch, best_val_loss)
        else:
            patience_counter += 1

        if (epoch + 1) % config.checkpoint_every == 0:
            periodic_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
            _save_checkpoint(
                periodic_path, model, optimizer, scheduler, epoch + 1,
                val_loss, val_metrics, estimation_config, config,
                loss_fn=loss_fn,
            )

        # -- Hard negative mining -----------------------------------------------
        if config.hard_negative_mining and epoch < config.num_epochs - 1:
            sample_weights = _compute_sample_weights(
                model, train_dataset, device,
                top_fraction=config.hard_negative_top_fraction,
                batch_size=config.batch_size,
            )
            sampler = WeightedRandomSampler(
                sample_weights, num_samples=len(sample_weights), replacement=True,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                sampler=sampler,
                num_workers=config.num_workers,
                pin_memory=use_pin_memory,
                persistent_workers=config.num_workers > 0,
            )
            n_hard = sum(1 for w in sample_weights if w > 1.0)
            logger.info("Hard negative mining: %d/%d samples upweighted", n_hard, len(sample_weights))

        # -- Early stopping ----------------------------------------------------
        if patience_counter >= config.early_stopping_patience:
            logger.info(
                "Early stopping at epoch %d (no improvement for %d epochs)",
                epoch + 1, config.early_stopping_patience,
            )
            break

    logger.info("Training complete. Best epoch: %d, best val_loss: %.6f", best_epoch, best_val_loss)

    # -- Upload to GCS ---------------------------------------------------------
    if config.gcs_checkpoint_bucket:
        _upload_to_gcs(best_path, config.gcs_checkpoint_bucket)

    # -- wandb finish ----------------------------------------------------------
    if wandb_run is not None:
        import wandb

        wandb.finish()

    return best_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    *,
    loss_type: str = "mse",
    direction_cosine_weight: float = 0.0,
    loss_fn: AnchoringLossWithUncertainty | None = None,
) -> tuple[float, dict[str, float]]:
    """Run validation and compute per-component metrics."""
    model.eval()
    loss_sum = 0.0
    count = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                preds = model(images)
                loss = _compute_loss(
                    preds, labels,
                    loss_type=loss_type,
                    direction_cosine_weight=direction_cosine_weight,
                    loss_fn=loss_fn,
                )

            loss_sum += loss.item() * images.size(0)
            count += images.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    val_loss = loss_sum / max(count, 1)

    # Per-component MAE
    preds_cat = torch.cat(all_preds, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    abs_err = (preds_cat - labels_cat).abs()

    metrics = {
        "oz_mae": abs_err[:, 2].mean().item(),     # AP position (most important)
        "u_mae": abs_err[:, 3:6].mean().item(),     # Width direction vector
        "v_mae": abs_err[:, 6:9].mean().item(),     # Height direction vector
        "origin_mae": abs_err[:, 0:3].mean().item(),
        "all_mae": abs_err.mean().item(),
    }
    return val_loss, metrics


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    val_loss: float,
    val_metrics: dict[str, float],
    estimation_config: EstimationConfig,
    training_config: TrainingConfig,
    loss_fn: AnchoringLossWithUncertainty | None = None,
) -> None:
    """Save an enriched checkpoint."""
    data: dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "val_loss": val_loss,
        "val_metrics": val_metrics,
        "config": estimation_config.model_dump(),
        "training_config": training_config.model_dump(),
    }
    if loss_fn is not None:
        data["loss_state_dict"] = loss_fn.state_dict()
    torch.save(data, str(path))


def _init_wandb(
    config: TrainingConfig,
    estimation_config: EstimationConfig,
    n_train: int,
    n_val: int,
) -> object | None:
    """Initialize wandb if available. Returns the run object or None."""
    try:
        import wandb

        run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            config={
                "training": config.model_dump(),
                "estimation": estimation_config.model_dump(),
                "n_train": n_train,
                "n_val": n_val,
            },
        )
        logger.info("wandb initialized: %s", run.url)
        return run
    except ImportError:
        logger.info("wandb not installed, skipping experiment tracking")
        return None
    except Exception as e:
        logger.warning("wandb init failed: %s (continuing without tracking)", e)
        return None


def _upload_to_gcs(local_path: Path, bucket_uri: str) -> None:
    """Upload a checkpoint to GCS."""
    bucket_uri = bucket_uri.rstrip("/")
    dest = f"{bucket_uri}/{local_path.name}"
    logger.info("Uploading checkpoint to %s", dest)
    try:
        subprocess.run(["gsutil", "cp", str(local_path), dest], check=True)
        logger.info("Upload complete: %s", dest)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning("GCS upload failed: %s (checkpoint saved locally at %s)", e, local_path)
