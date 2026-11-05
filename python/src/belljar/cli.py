"""Command-line interface for the belljar pipeline.

Provides subcommands for each pipeline step, a full pipeline run,
and the JSON-RPC server. Every subcommand builds the same params model
the JSON-RPC server builds, so the two entry points validate identically.

Usage:
    belljar estimate --input-dir SECTIONS --output-dir OUT --model-path model.pt
    belljar align --input-dir SECTIONS --output-dir OUT --estimates OUT/estimates.json
    belljar detect --input-dir SECTIONS --output-dir OUT --model-path yolo.pt
    belljar count --predictions-dir DETECTIONS --annotations-dir ALIGNED --output-dir OUT
    belljar collate --input-csv OUT/count_results.csv --output-dir OUT
    belljar run --input-dir SECTIONS --output-dir OUT --model-path model.pt \\
        --detection-model yolo.pt
    belljar server
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from belljar.config import BelljarConfig
from belljar.pipeline.params import (
    AlignParams,
    CollateParams,
    CountParams,
    DetectParams,
    EstimateParams,
    MaxProjectionParams,
    SharpenParams,
    StepParams,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from belljar.pipeline.base import PipelineStep
    from belljar.server import BelljarServer

logger = logging.getLogger(__name__)


def _progress(current: int, total: int, message: str) -> None:
    """CLI progress callback — prints to stderr."""
    if total > 0:
        pct = current / total * 100
        sys.stderr.write(f"\r[{pct:5.1f}%] {message}")
        if current >= total:
            sys.stderr.write("\n")
        sys.stderr.flush()


class InputError(Exception):
    """An input file could not be read or does not have the expected shape."""


def _load_alignments(estimates_path: Path) -> list[dict[str, Any]]:
    try:
        with open(estimates_path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        raise InputError(f"Could not read {estimates_path}: {e}") from e
    if not isinstance(data, dict) or "alignments" not in data:
        raise InputError(f"{estimates_path} has no 'alignments' list")
    return data["alignments"]


def _max_projection_params(args: argparse.Namespace) -> StepParams:
    return MaxProjectionParams(input_dir=args.input_dir, output_dir=args.output_dir)


def _sharpen_params(args: argparse.Namespace) -> StepParams:
    return SharpenParams(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        equalize=args.equalize,
        radius=args.radius,
        amount=args.amount,
    )


def _estimate_params(args: argparse.Namespace) -> StepParams:
    return EstimateParams(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        gcs_model_uri=args.gcs_model_uri,
        uncertainty=args.uncertainty,
    )


def _align_params(args: argparse.Namespace) -> StepParams:
    return _align_params_from(args.input_dir, args.output_dir, args.estimates)


def _align_params_from(input_dir: Path, output_dir: Path, estimates: Path) -> AlignParams:
    return AlignParams.model_validate(
        {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "alignments": _load_alignments(estimates),
        }
    )


def _detect_params(args: argparse.Namespace) -> StepParams:
    return DetectParams(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        save_visualizations=not args.no_visualizations,
    )


def _count_params(args: argparse.Namespace) -> StepParams:
    return CountParams(
        predictions_dir=args.predictions_dir,
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir,
        include_layers=args.include_layers,
    )


def _collate_params(args: argparse.Namespace) -> StepParams:
    return CollateParams(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        include_layers=args.include_layers,
    )


_PARAM_BUILDERS: dict[str, Callable[[argparse.Namespace], StepParams]] = {
    "max_projection": _max_projection_params,
    "sharpen": _sharpen_params,
    "estimate": _estimate_params,
    "align": _align_params,
    "detect": _detect_params,
    "count": _count_params,
    "collate": _collate_params,
}


def build_step_params(args: argparse.Namespace) -> StepParams:
    """Build the params model for the subcommand in ``args``."""
    return _PARAM_BUILDERS[args.command](args)


def _make_step(step_name: str, config: BelljarConfig) -> PipelineStep[Any]:
    from belljar.server import STEP_CLASSES

    return STEP_CLASSES[step_name](config)


def _execute(step: PipelineStep[Any], params: StepParams) -> bool:
    """Validate then run one step, logging the outcome."""
    errors = step.validate_inputs(params)
    if errors:
        for err in errors:
            logger.error("Validation error: %s", err)
        return False

    result = step.run(_progress, params)

    if result.success:
        logger.info("Step '%s' completed successfully", step.name)
        if result.output_path:
            logger.info("Output: %s", result.output_path)
        if result.metrics:
            logger.info("Metrics: %s", json.dumps(result.metrics, indent=2))
    else:
        logger.error("Step '%s' failed", step.name)
        for err in result.errors:
            logger.error("  %s", err)

    for w in result.warnings:
        logger.warning("  %s", w)

    return result.success


def _run_step(args: argparse.Namespace, config: BelljarConfig) -> int:
    """Run a single pipeline step from CLI args."""
    try:
        params = build_step_params(args)
    except ValidationError as e:
        logger.error("Invalid arguments: %s", e)
        return 1
    except InputError as e:
        logger.error("%s", e)
        return 1
    return 0 if _execute(_make_step(args.command, config), params) else 1


def _run_full_pipeline(args: argparse.Namespace, config: BelljarConfig) -> int:
    """Run estimate -> align -> detect -> count -> collate; 1 on any failure."""
    try:
        return _run_pipeline_steps(args, config)
    except ValidationError as e:
        logger.error("Invalid arguments: %s", e)
        return 1
    except InputError as e:
        logger.error("%s", e)
        return 1


def _run_pipeline_steps(args: argparse.Namespace, config: BelljarConfig) -> int:
    """``--detections-dir`` reuses existing detection results in place of detect."""
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    estimates_dir = output_dir / "estimates"
    aligned_dir = output_dir / "aligned"
    detections_dir: Path = args.detections_dir or output_dir / "detections"
    counts_dir = output_dir / "counts"
    results_dir = output_dir / "results"

    logger.info("=== Step 1/5: Estimate ===")
    estimate = EstimateParams(
        input_dir=input_dir,
        output_dir=estimates_dir,
        model_path=args.model_path,
        gcs_model_uri=args.gcs_model_uri,
        uncertainty=args.uncertainty,
    )
    if not _execute(_make_step("estimate", config), estimate):
        return 1

    logger.info("=== Step 2/5: Align ===")
    align = _align_params_from(input_dir, aligned_dir, estimates_dir / "estimates.json")
    if not _execute(_make_step("align", config), align):
        return 1

    if args.detections_dir is None:
        logger.info("=== Step 3/5: Detect ===")
        detect = DetectParams(
            input_dir=input_dir,
            output_dir=detections_dir,
            model_path=args.detection_model,
        )
        if not _execute(_make_step("detect", config), detect):
            return 1
    else:
        logger.info("=== Step 3/5: Detect (using %s) ===", detections_dir)

    logger.info("=== Step 4/5: Count ===")
    count = CountParams(
        predictions_dir=detections_dir,
        annotations_dir=aligned_dir,
        output_dir=counts_dir,
        include_layers=args.include_layers,
    )
    if not _execute(_make_step("count", config), count):
        return 1

    logger.info("=== Step 5/5: Collate ===")
    collate = CollateParams(
        input_csv=counts_dir / "count_results.csv",
        output_dir=results_dir,
        include_layers=args.include_layers,
    )
    if not _execute(_make_step("collate", config), collate):
        return 1

    logger.info("=== Pipeline complete ===")
    logger.info("Results: %s", results_dir)
    return 0


def _build_server(args: argparse.Namespace) -> BelljarServer:
    from belljar.server import BelljarServer

    return BelljarServer(_load_config(args.config))


def _run_server(args: argparse.Namespace) -> int:
    """Start the JSON-RPC server on stdio."""
    try:
        server = _build_server(args)
    except (OSError, ValidationError) as e:
        logger.error("Could not load config %s: %s", args.config, e)
        return 1
    server.run()
    return 0


def _add_io_dirs(parser: argparse.ArgumentParser, input_help: str, output_help: str) -> None:
    parser.add_argument("--input-dir", type=Path, required=True, help=input_help)
    parser.add_argument("--output-dir", type=Path, required=True, help=output_help)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="belljar",
        description="Belljar: automatic alignment tool for mouse brain histology.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="JSON file of BelljarConfig overrides (default: built-in defaults).",
    )
    subparsers = parser.add_subparsers(dest="command")

    sections_help = "Directory of tissue section images."

    mxp = subparsers.add_parser("max_projection", help="Max-project z-stacks to 2D images.")
    _add_io_dirs(mxp, "Directory of z-stack TIFFs.", "Output directory for projected images.")

    shp = subparsers.add_parser("sharpen", help="Sharpen tissue images.")
    _add_io_dirs(shp, sections_help, "Output directory for sharpened images.")
    shp.add_argument("--equalize", action="store_true", help="Apply CLAHE before sharpening.")
    shp.add_argument("--radius", type=float, default=3.0, help="Unsharp mask radius.")
    shp.add_argument("--amount", type=float, default=2.0, help="Unsharp mask amount.")

    est = subparsers.add_parser("estimate", help="Estimate slice positions and angles.")
    _add_io_dirs(est, sections_help, "Output directory for estimates JSON.")
    est.add_argument(
        "--model-path", type=Path, default=None, help="Path to estimator model checkpoint."
    )
    est.add_argument(
        "--gcs-model-uri", type=str, default=None, help="GCS URI to download model from."
    )
    est.add_argument(
        "--uncertainty", action="store_true", help="Enable MC Dropout uncertainty estimation."
    )

    aln = subparsers.add_parser("align", help="Register tissue sections to atlas.")
    _add_io_dirs(aln, sections_help, "Output directory for registration results.")
    aln.add_argument(
        "--estimates", type=Path, required=True, help="Path to estimates.json from estimate step."
    )

    det = subparsers.add_parser("detect", help="Detect cells in tissue sections.")
    _add_io_dirs(det, sections_help, "Output directory for detection results.")
    det.add_argument(
        "--model-path", type=Path, required=True, help="Path to the YOLO detection model."
    )
    det.add_argument(
        "--no-visualizations", action="store_true", help="Skip writing detection overlay PNGs."
    )

    cnt = subparsers.add_parser("count", help="Count detected cells per brain region.")
    cnt.add_argument(
        "--predictions-dir", type=Path, required=True, help="Directory of detection pickles."
    )
    cnt.add_argument(
        "--annotations-dir",
        type=Path,
        required=True,
        help="Directory of Annotation_<section>.npy/.pkl files from the align step.",
    )
    cnt.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory for count results."
    )
    cnt.add_argument(
        "--include-layers", action="store_true", help="Count cortical layers separately."
    )

    col = subparsers.add_parser("collate", help="Collate counts into summary CSV.")
    col.add_argument(
        "--input-csv", type=Path, required=True, help="count_results.csv from the count step."
    )
    col.add_argument(
        "--output-dir", type=Path, required=True, help="Output directory for summary CSV."
    )
    col.add_argument(
        "--include-layers", action="store_true", help="Keep cortical layers as separate rows."
    )

    run = subparsers.add_parser(
        "run", help="Run the full pipeline (estimate -> align -> detect -> count -> collate)."
    )
    _add_io_dirs(run, sections_help, "Output directory for all results.")
    run.add_argument(
        "--model-path", type=Path, default=None, help="Path to estimator model checkpoint."
    )
    run.add_argument(
        "--gcs-model-uri", type=str, default=None, help="GCS URI to download the estimator from."
    )
    run.add_argument(
        "--uncertainty", action="store_true", help="Enable MC Dropout uncertainty estimation."
    )
    detection_source = run.add_mutually_exclusive_group(required=True)
    detection_source.add_argument(
        "--detection-model", type=Path, help="Path to the YOLO detection model."
    )
    detection_source.add_argument(
        "--detections-dir", type=Path, help="Existing detection pickles; skips the detect step."
    )
    run.add_argument(
        "--include-layers", action="store_true", help="Count cortical layers separately."
    )

    subparsers.add_parser("server", help="Start the JSON-RPC server (for Electron frontend).")

    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return _build_parser().parse_args(argv)


def _load_config(path: Path | None) -> BelljarConfig:
    if path is None:
        return BelljarConfig()
    return BelljarConfig.load(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

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

    try:
        config = _load_config(args.config)
    except (OSError, ValidationError) as e:
        logger.error("Could not load config %s: %s", args.config, e)
        return 1

    if args.command == "run":
        return _run_full_pipeline(args, config)

    return _run_step(args, config)


if __name__ == "__main__":
    sys.exit(main())
