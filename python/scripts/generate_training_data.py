#!/usr/bin/env python3
"""CLI for belljar training data generation.

Wraps generate_dataset() with argument parsing, atlas pre-caching,
timing, and optional GCS upload.

Usage:
    # Local generation (small test)
    python scripts/generate_training_data.py --output-dir /tmp/test --num-samples 100

    # Full generation with GCS upload
    python scripts/generate_training_data.py \
        --output-dir /data/training \
        --num-samples 100000 \
        --num-workers 8 \
        --gcs-bucket gs://belljar-training-data \
        --compress
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path

logger = logging.getLogger("belljar.datagen")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate belljar training data (synthetic atlas slices with anchoring labels).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Local output directory for PNGs and metadata.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100_000,
        help="Number of training samples to generate (default: 100000).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel workers. Default: all available CPUs, capped at 32.",
    )
    parser.add_argument(
        "--atlas-name",
        type=str,
        default="allen_mouse_10um",
        help="BrainGlobe atlas identifier (default: allen_mouse_10um).",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="default",
        help="Reference modality: 'default' (STP) or 'nissl' (default: default).",
    )
    parser.add_argument(
        "--gcs-bucket",
        type=str,
        default=None,
        help="GCS bucket URI for upload after generation (e.g. gs://my-bucket).",
    )
    parser.add_argument(
        "--gcs-prefix",
        type=str,
        default="datasets/",
        help="Prefix within GCS bucket (default: datasets/).",
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        help="Compress output to .tar.gz before uploading to GCS.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master RNG seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def ensure_atlas(atlas_name: str, reference_name: str = "default") -> None:
    """Pre-download the atlas so workers don't race on first load."""
    from belljar.atlas.provider import AtlasProvider

    logger.info("Pre-caching atlas: %s (reference: %s)", atlas_name, reference_name)
    provider = AtlasProvider(atlas_name, reference_name=reference_name)
    _ = provider.reference  # triggers download if not cached
    logger.info("Atlas cached. Shape: %s", provider.shape)


def get_dir_size_mb(path: Path) -> float:
    """Get total size of a directory in MB."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def upload_to_gcs(local_path: Path, bucket: str, prefix: str, compress: bool) -> None:
    """Upload output directory to GCS using gsutil."""
    # Normalize bucket URI
    bucket = bucket.rstrip("/")
    prefix = prefix.strip("/")

    if compress:
        archive = local_path.parent / f"{local_path.name}.tar.gz"
        logger.info("Compressing to %s ...", archive)
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(local_path, arcname=local_path.name)
        archive_mb = archive.stat().st_size / (1024 * 1024)
        logger.info("Archive size: %.1f MB", archive_mb)

        dest = f"{bucket}/{prefix}/{archive.name}"
        cmd = ["gsutil", "-m", "cp", str(archive), dest]
    else:
        dest = f"{bucket}/{prefix}/{local_path.name}/"
        cmd = ["gsutil", "-m", "rsync", "-r", str(local_path), dest]

    logger.info("Uploading: %s -> %s", local_path, dest)
    subprocess.run(cmd, check=True)
    logger.info("Upload complete: %s", dest)


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Auto-detect worker count if not specified
    if args.num_workers is None:
        cpu_count = os.cpu_count() or 4
        mem_gb = 0.0
        # Each worker needs ~200 MB working memory (atlas is shared via memmap).
        # Reserve ~4 GB for OS + main process, use the rest.
        try:
            mem_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            mem_gb = mem_bytes / (1024**3)
            max_by_mem = max(1, int((mem_gb - 4) / 0.2))
        except (ValueError, OSError):
            max_by_mem = 32  # can't detect memory, use CPU-based cap
        args.num_workers = min(cpu_count, max_by_mem, 32)
        logger.info("Auto-detected %d CPUs, %.0f GB RAM -> using %d workers",
                     cpu_count, mem_gb, args.num_workers)

    logger.info(
        "Configuration: samples=%d, workers=%d, atlas=%s, reference=%s, seed=%d",
        args.num_samples,
        args.num_workers,
        args.atlas_name,
        args.reference,
        args.seed,
    )

    # Step 1: Pre-cache atlas (single download, avoids worker races)
    try:
        ensure_atlas(args.atlas_name, reference_name=args.reference)
    except Exception as e:
        logger.error("Failed to load atlas '%s' (reference=%s): %s", args.atlas_name, args.reference, e)
        return 1

    # Step 2: Generate dataset
    from belljar.config import DataGenerationConfig
    from belljar.estimation.data_generation import generate_dataset

    config = DataGenerationConfig(num_samples=args.num_samples)

    t0 = time.time()
    try:
        output_path = generate_dataset(
            output_dir=args.output_dir,
            atlas_name=args.atlas_name,
            config=config,
            num_workers=args.num_workers,
            seed=args.seed,
            reference_name=args.reference,
        )
    except Exception as e:
        logger.error("Generation failed: %s", e)
        return 1

    elapsed = time.time() - t0
    size_mb = get_dir_size_mb(output_path)
    png_count = len(list(output_path.glob("*.png")))

    logger.info(
        "Generation complete: %d PNGs, %.1f MB, %.1f sec (%.0f samples/sec)",
        png_count,
        size_mb,
        elapsed,
        png_count / elapsed if elapsed > 0 else 0,
    )

    # Step 3: Upload to GCS if requested
    if args.gcs_bucket:
        try:
            upload_to_gcs(output_path, args.gcs_bucket, args.gcs_prefix, args.compress)
        except subprocess.CalledProcessError as e:
            logger.error("GCS upload failed: %s", e)
            logger.info("Data remains at: %s", output_path)
            return 2

    logger.info("Done. Output: %s", output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
