#!/usr/bin/env python3
"""CLI for training the belljar slice position estimator.

Wraps the training loop with argument parsing, device detection, and logging.

Usage:
    # Local quick test
    python scripts/train_estimator.py --data-dir /tmp/test --output-dir /tmp/model --epochs 5

    # GCP T4 full training
    python scripts/train_estimator.py \
        --data-dir /data/training \
        --output-dir /data/checkpoints \
        --epochs 50 --batch-size 128 \
        --gcs-bucket gs://belljar-training-data/checkpoints \
        --wandb-project belljar-estimator

    # Resume from checkpoint
    python scripts/train_estimator.py \
        --data-dir /data/training \
        --output-dir /data/checkpoints \
        --resume /data/checkpoints/checkpoint_epoch_20.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("belljar.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the belljar slice position estimator (ResNet50 → 9 anchoring values).",
    )
    parser.add_argument(
        "--data-dir", type=Path, required=True,
        help="Training data directory (PNGs + metadata.pkl).",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for checkpoints.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Max training epochs (default: 50).")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate (default: 1e-3).")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Linear warmup epochs.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument(
        "--no-amp", action="store_true",
        help="Disable mixed precision (AMP). Useful for CPU or debugging.",
    )
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience.")
    parser.add_argument(
        "--gcs-bucket", type=str, default=None,
        help="GCS bucket URI for checkpoint upload (e.g. gs://my-bucket/checkpoints).",
    )
    parser.add_argument("--wandb-project", type=str, default="belljar-estimator", help="W&B project.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="W&B entity/team.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")
    parser.add_argument(
        "--resume", type=Path, default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Validate data dir
    if not args.data_dir.exists():
        logger.error("Data directory does not exist: %s", args.data_dir)
        return 1

    metadata_path = args.data_dir / "metadata.pkl"
    if not metadata_path.exists():
        logger.error("No metadata.pkl found in %s", args.data_dir)
        return 1

    png_count = len(list(args.data_dir.glob("*.png")))
    logger.info("Found %d training images in %s", png_count, args.data_dir)

    if png_count == 0:
        logger.error("No PNG files found in data directory")
        return 1

    # Build configs
    from belljar.config import EstimationConfig, TrainingConfig

    training_config = TrainingConfig(
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        num_workers=args.num_workers,
        mixed_precision=not args.no_amp,
        early_stopping_patience=args.patience,
        gcs_checkpoint_bucket=args.gcs_bucket,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        seed=args.seed,
    )
    estimation_config = EstimationConfig()

    # Train
    from belljar.estimation.train import train

    try:
        best_path = train(
            data_dir=args.data_dir,
            config=training_config,
            estimation_config=estimation_config,
            output_dir=args.output_dir,
        )
        logger.info("Training complete. Best model: %s", best_path)
        return 0
    except Exception as e:
        logger.error("Training failed: %s", e, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
