#!/usr/bin/env python3
"""CLI for downloading Allen Institute ISH section images with alignment labels.

Queries the Allen Brain Atlas RMA API for coronal ISH experiments, filters
by alignment quality, downloads section images, computes 9-value anchoring
labels from alignment transforms, and saves everything for training.

Usage:
    # Smoke test (10 experiments)
    python scripts/download_allen_data.py --output-dir /tmp/allen --smoketest

    # Full download
    python scripts/download_allen_data.py --output-dir /data/allen_ish --workers 4

    # Resume interrupted download
    python scripts/download_allen_data.py --output-dir /data/allen_ish --resume
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger("belljar.allen_download")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Allen Institute ISH section images with alignment labels.",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True,
        help="Output directory for downloaded images and metadata.",
    )
    parser.add_argument(
        "--product-id", type=int, default=1,
        help="Allen product ID (1=ISH, 5=Connectivity). Default: 1.",
    )
    parser.add_argument(
        "--max-experiments", type=int, default=None,
        help="Maximum number of experiments to download (None=all).",
    )
    parser.add_argument(
        "--downsample", type=int, default=4,
        help="Image downsample level (0=full, 4=16x smaller). Default: 4.",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Number of parallel download workers. Default: 4.",
    )
    parser.add_argument(
        "--min-quality", type=float, default=0.7,
        help="Minimum quality score to accept an experiment. Default: 0.7.",
    )
    parser.add_argument(
        "--rate-limit", type=float, default=0.5,
        help="Seconds between API/download requests. Default: 0.5.",
    )
    parser.add_argument(
        "--smoketest", action="store_true",
        help="Quick test mode: download only 10 experiments.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-downloaded images.",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def _download_experiment_images(
    experiment: dict,
    output_dir: Path,
    downsample: int,
    rate_limit: float,
    resume: bool,
) -> dict:
    """Download images for a single experiment and compute labels.

    Returns metadata dict for this experiment, or empty dict on failure.
    """
    from belljar.estimation.allen_data import (
        compose_alignment_transforms,
        download_section_image,
        get_section_images,
    )

    exp_id = experiment.get("id", "unknown")
    exp_dir = output_dir / str(exp_id)

    sections = get_section_images(experiment)
    if not sections:
        logger.debug("Experiment %s has no section images", exp_id)
        return {}

    exp_meta: dict = {"sections": {}}
    downloaded = 0
    skipped = 0

    for section in sections:
        sec_id = section.get("id")
        if sec_id is None:
            continue

        image_path = exp_dir / f"{sec_id}.png"

        # Compute anchoring label
        anchoring = compose_alignment_transforms(section, experiment)
        if anchoring is None:
            continue

        # Download image (or skip if resuming)
        if resume and image_path.exists():
            skipped += 1
        else:
            success = download_section_image(
                section_image_id=sec_id,
                output_path=image_path,
                downsample=downsample,
                rate_limit=rate_limit,
            )
            if not success:
                continue
            downloaded += 1

        exp_meta["sections"][str(sec_id)] = {
            "anchoring": anchoring.tolist(),
            "experiment_id": exp_id,
            "section_id": sec_id,
        }

    logger.info(
        "Experiment %s: %d downloaded, %d skipped, %d labeled",
        exp_id, downloaded, skipped, len(exp_meta["sections"]),
    )
    return exp_meta


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.smoketest:
        args.max_experiments = 10
        logger.info("Smoke test mode: limiting to %d experiments", args.max_experiments)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Query experiments
    from belljar.estimation.allen_data import (
        assess_experiment_quality,
        get_section_images,
        query_allen_experiments,
    )

    logger.info("Querying Allen API for experiments (product_id=%d)...", args.product_id)
    experiments = query_allen_experiments(
        product_id=args.product_id,
        rate_limit=args.rate_limit,
        max_rows=args.max_experiments,
    )
    logger.info("Fetched %d experiments", len(experiments))

    # Filter by quality
    logger.info("Filtering experiments by quality (min_score=%.2f)...", args.min_quality)
    filtered = []
    for exp in experiments:
        sections = get_section_images(exp)
        score = assess_experiment_quality(sections, exp)
        if score >= args.min_quality:
            filtered.append(exp)

    logger.info("Quality filter: %d/%d experiments passed", len(filtered), len(experiments))

    if not filtered:
        logger.warning("No experiments passed quality filter")
        return 0

    # Download images and compute labels
    all_metadata: dict = {}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                _download_experiment_images,
                exp,
                args.output_dir,
                args.downsample,
                args.rate_limit,
                args.resume,
            ): exp.get("id", "unknown")
            for exp in filtered
        }

        for i, future in enumerate(as_completed(futures), 1):
            exp_id = futures[future]
            try:
                exp_meta = future.result()
                if exp_meta:
                    all_metadata[str(exp_id)] = exp_meta
            except Exception as e:
                logger.warning("Experiment %s failed: %s", exp_id, e)

            if i % 10 == 0 or i == len(futures):
                logger.info("Progress: %d/%d experiments processed", i, len(futures))

    # Save metadata
    metadata_path = args.output_dir / "metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(all_metadata, f)

    total_sections = sum(
        len(exp.get("sections", {})) for exp in all_metadata.values()
    )
    logger.info(
        "Download complete: %d experiments, %d sections, metadata saved to %s",
        len(all_metadata), total_sections, metadata_path,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
