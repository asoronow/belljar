"""Command-line interface for the belljar pipeline.

Provides subcommands for each pipeline step, a full pipeline run,
and the JSON-RPC server.

Usage:
    belljar estimate --input-dir /path/to/sections --output-dir /path/to/output --model-path model.pt
    belljar align --input-dir /path/to/sections --output-dir /path/to/output --estimates /path/to/estimates.json
    belljar detect --input-dir /path/to/sections --output-dir /path/to/output
    belljar count --input-dir /path/to/detections --output-dir /path/to/output
    belljar collate --input-dir /path/to/counts --output-dir /path/to/output
    belljar run --input-dir /path/to/sections --output-dir /path/to/output --model-path model.pt
    belljar server
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _progress(current: int, total: int, message: str) -> None:
    """CLI progress callback — prints to stderr."""
    if total > 0:
        pct = current / total * 100
        sys.stderr.write(f"\r[{pct:5.1f}%] {message}")
        if current >= total:
            sys.stderr.write("\n")
        sys.stderr.flush()


def _run_step(step_name: str, args: argparse.Namespace) -> int:
    """Run a single pipeline step from CLI args."""
    from belljar.config import BelljarConfig

    config = BelljarConfig()

    step_kwargs: dict[str, Any] = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
    }

    # Step-specific arguments
    if step_name == "estimate":
        from belljar.pipeline.estimate import EstimateStep

        if args.model_path:
            step_kwargs["model_path"] = str(args.model_path)
        if hasattr(args, "gcs_model_uri") and args.gcs_model_uri:
            step_kwargs["gcs_model_uri"] = args.gcs_model_uri
        if hasattr(args, "uncertainty") and args.uncertainty:
            step_kwargs["uncertainty"] = True
        step = EstimateStep(config)

    elif step_name == "align":
        from belljar.pipeline.align import AlignStep

        estimates_path = Path(args.estimates)
        if not estimates_path.exists():
            logger.error("Estimates file not found: %s", estimates_path)
            return 1
        with open(estimates_path) as f:
            data = json.load(f)
        step_kwargs["alignments"] = data["alignments"]
        step = AlignStep(config)

    elif step_name == "detect":
        from belljar.pipeline.detect import DetectStep

        step = DetectStep(config)

    elif step_name == "count":
        from belljar.pipeline.count import CountStep

        step = CountStep(config)

    elif step_name == "collate":
        from belljar.pipeline.collate import CollateStep

        step = CollateStep(config)

    else:
        logger.error("Unknown step: %s", step_name)
        return 1

    # Validate
    errors = step.validate_inputs(**step_kwargs)
    if errors:
        for err in errors:
            logger.error("Validation error: %s", err)
        return 1

    # Run
    result = step.run(_progress, **step_kwargs)

    if result.success:
        logger.info("Step '%s' completed successfully", step_name)
        if result.output_path:
            logger.info("Output: %s", result.output_path)
        if result.metrics:
            logger.info("Metrics: %s", json.dumps(result.metrics, indent=2))
    else:
        logger.error("Step '%s' failed", step_name)
        for err in result.errors:
            logger.error("  %s", err)

    for w in result.warnings:
        logger.warning("  %s", w)

    return 0 if result.success else 1


def _run_full_pipeline(args: argparse.Namespace) -> int:
    """Run the full pipeline: estimate -> align -> detect -> count -> collate."""
    from belljar.config import BelljarConfig
    from belljar.pipeline.align import AlignStep
    from belljar.pipeline.collate import CollateStep
    from belljar.pipeline.count import CountStep
    from belljar.pipeline.detect import DetectStep
    from belljar.pipeline.estimate import EstimateStep

    config = BelljarConfig()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps = [
        ("estimate", EstimateStep(config), {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir / "estimates"),
            "model_path": str(args.model_path) if args.model_path else None,
        }),
    ]

    # Run estimate
    logger.info("=== Step 1/5: Estimate ===")
    est_result = steps[0][1].run(_progress, **steps[0][2])
    if not est_result.success:
        logger.error("Estimation failed: %s", est_result.errors)
        return 1

    # Load estimates for alignment
    estimates_path = output_dir / "estimates" / "estimates.json"
    with open(estimates_path) as f:
        estimates_data = json.load(f)

    # Align
    logger.info("=== Step 2/5: Align ===")
    align_step = AlignStep(config)
    align_result = align_step.run(_progress, **{
        "input_dir": str(input_dir),
        "output_dir": str(output_dir / "aligned"),
        "alignments": estimates_data["alignments"],
    })
    if not align_result.success:
        logger.error("Alignment failed: %s", align_result.errors)
        return 1

    # Detect
    logger.info("=== Step 3/5: Detect ===")
    detect_step = DetectStep(config)
    detect_result = detect_step.run(_progress, **{
        "input_dir": str(input_dir),
        "output_dir": str(output_dir / "detections"),
    })
    if not detect_result.success:
        logger.error("Detection failed: %s", detect_result.errors)
        return 1

    # Count
    logger.info("=== Step 4/5: Count ===")
    count_step = CountStep(config)
    count_result = count_step.run(_progress, **{
        "input_dir": str(output_dir / "detections"),
        "output_dir": str(output_dir / "counts"),
    })
    if not count_result.success:
        logger.error("Counting failed: %s", count_result.errors)
        return 1

    # Collate
    logger.info("=== Step 5/5: Collate ===")
    collate_step = CollateStep(config)
    collate_result = collate_step.run(_progress, **{
        "input_dir": str(output_dir / "counts"),
        "output_dir": str(output_dir / "results"),
    })
    if not collate_result.success:
        logger.error("Collation failed: %s", collate_result.errors)
        return 1

    logger.info("=== Pipeline complete ===")
    logger.info("Results: %s", output_dir / "results")
    return 0


def _run_server(args: argparse.Namespace) -> int:
    """Start the JSON-RPC server."""
    from belljar.server import main as server_main

    server_main()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="belljar",
        description="Belljar: automatic alignment tool for mouse brain histology.",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── estimate ──────────────────────────────────────────────────────
    est = subparsers.add_parser("estimate", help="Estimate slice positions and angles.")
    est.add_argument("--input-dir", type=Path, required=True, help="Directory of tissue section images.")
    est.add_argument("--output-dir", type=Path, required=True, help="Output directory for estimates JSON.")
    est.add_argument("--model-path", type=Path, default=None, help="Path to estimator model checkpoint.")
    est.add_argument("--gcs-model-uri", type=str, default=None, help="GCS URI to download model from.")
    est.add_argument("--uncertainty", action="store_true", help="Enable MC Dropout uncertainty estimation.")

    # ── align ─────────────────────────────────────────────────────────
    aln = subparsers.add_parser("align", help="Register tissue sections to atlas.")
    aln.add_argument("--input-dir", type=Path, required=True, help="Directory of tissue section images.")
    aln.add_argument("--output-dir", type=Path, required=True, help="Output directory for registration results.")
    aln.add_argument("--estimates", type=Path, required=True, help="Path to estimates.json from estimate step.")

    # ── detect ────────────────────────────────────────────────────────
    det = subparsers.add_parser("detect", help="Detect cells in tissue sections.")
    det.add_argument("--input-dir", type=Path, required=True, help="Directory of tissue section images.")
    det.add_argument("--output-dir", type=Path, required=True, help="Output directory for detection results.")

    # ── count ─────────────────────────────────────────────────────────
    cnt = subparsers.add_parser("count", help="Count detected cells per brain region.")
    cnt.add_argument("--input-dir", type=Path, required=True, help="Directory of detection results.")
    cnt.add_argument("--output-dir", type=Path, required=True, help="Output directory for count results.")

    # ── collate ───────────────────────────────────────────────────────
    col = subparsers.add_parser("collate", help="Collate counts into summary CSV.")
    col.add_argument("--input-dir", type=Path, required=True, help="Directory of count results.")
    col.add_argument("--output-dir", type=Path, required=True, help="Output directory for summary CSV.")

    # ── run ────────────────────────────────────────────────────────────
    run = subparsers.add_parser("run", help="Run the full pipeline (estimate → align → detect → count → collate).")
    run.add_argument("--input-dir", type=Path, required=True, help="Directory of tissue section images.")
    run.add_argument("--output-dir", type=Path, required=True, help="Output directory for all results.")
    run.add_argument("--model-path", type=Path, default=None, help="Path to estimator model checkpoint.")

    # ── server ────────────────────────────────────────────────────────
    subparsers.add_parser("server", help="Start the JSON-RPC server (for Electron frontend).")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "server":
        return _run_server(args)

    if args.command == "run":
        return _run_full_pipeline(args)

    # Single step commands
    return _run_step(args.command, args)


if __name__ == "__main__":
    sys.exit(main())
