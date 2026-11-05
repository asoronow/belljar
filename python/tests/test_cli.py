"""Tests for the belljar command-line interface."""

import json
import pickle
from pathlib import Path
from unittest.mock import patch

import cv2
import pytest
import torch

from belljar.cli import build_step_params, main, parse_args
from belljar.config import BelljarConfig
from belljar.pipeline.params import CountParams, DetectParams
from belljar.types import DetectionResult

CONSTANT_OUTPUT = [0.5, 0.5, 0.05, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
STEMS = ["sec_a", "sec_b"]


@pytest.fixture
def constant_model(tmp_path: Path) -> Path:
    from belljar.estimation.predictor import SliceEstimator

    model = SliceEstimator(num_outputs=9, dropout_rate=0.2, orthogonalize=True)
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.copy_(torch.tensor(CONSTANT_OUTPUT))
    model_path = tmp_path / "constant_model.pt"
    torch.save({"model_state_dict": model.state_dict()}, str(model_path))
    return model_path


@pytest.fixture
def sections(tmp_path: Path, structured_tissue) -> Path:
    img_dir = tmp_path / "sections"
    img_dir.mkdir()
    for stem in STEMS:
        cv2.imwrite(str(img_dir / f"{stem}.png"), structured_tissue)
    return img_dir


@pytest.fixture
def detections(tmp_path: Path) -> Path:
    pred_dir = tmp_path / "detections"
    pred_dir.mkdir()
    for stem in STEMS:
        result = DetectionResult(
            boxes=[[10.0, 10.0, 20.0, 20.0], [100.0, 200.0, 110.0, 210.0]],
            scores=[0.9, 0.9],
            image_width=256,
            image_height=256,
        )
        with open(pred_dir / f"{stem}.pkl", "wb") as f:
            pickle.dump([result], f)
    return pred_dir


@pytest.fixture
def fast_config(tmp_path: Path) -> Path:
    config = BelljarConfig()
    config.registration.processing_resolution = 128
    config.registration.rigid_iterations = 5
    config.registration.affine_iterations = 5
    config.registration.bspline_iterations = 5
    config.registration.bspline_grid_size = 3
    path = tmp_path / "config.json"
    config.save(path)
    return path


class TestParamConstruction:
    def test_count_args_build_count_params(self) -> None:
        args = parse_args(
            [
                "count",
                "--predictions-dir", "/p",
                "--annotations-dir", "/a",
                "--output-dir", "/o",
                "--include-layers",
            ]
        )
        params = build_step_params(args)
        assert params == CountParams(
            predictions_dir="/p", annotations_dir="/a", output_dir="/o", include_layers=True
        )

    def test_detect_args_build_detect_params(self) -> None:
        args = parse_args(
            ["detect", "--input-dir", "/i", "--output-dir", "/o", "--model-path", "/m.pt"]
        )
        params = build_step_params(args)
        assert params == DetectParams(input_dir="/i", output_dir="/o", model_path="/m.pt")
        assert params.save_visualizations is True

    def test_detect_requires_model_path(self, capsys) -> None:
        with pytest.raises(SystemExit):
            parse_args(["detect", "--input-dir", "/i", "--output-dir", "/o"])
        assert "--model-path" in capsys.readouterr().err

    def test_count_missing_dir_fails_validation(self, tmp_path: Path) -> None:
        code = main(
            [
                "count",
                "--predictions-dir", str(tmp_path / "nope"),
                "--annotations-dir", str(tmp_path),
                "--output-dir", str(tmp_path / "out"),
            ]
        )
        assert code == 1

    def test_stale_config_key_is_an_error(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"registration": {"rigid_learning_rate": 1.0}}))
        code = main(["--config", str(path), "count", "--predictions-dir", "/p",
                     "--annotations-dir", "/a", "--output-dir", "/o"])
        assert code == 1

class TestServerConfig:
    def test_server_subcommand_uses_config_file(self, tmp_path: Path) -> None:
        from belljar.cli import _build_server

        config = BelljarConfig()
        config.registration.rigid_iterations = 7
        path = tmp_path / "config.json"
        config.save(path)

        server = _build_server(parse_args(["--config", str(path), "server"]))
        assert server._get_config()["registration"]["rigid_iterations"] == 7

    def test_server_subcommand_rejects_bad_config(self, tmp_path: Path) -> None:
        path = tmp_path / "config.json"
        path.write_text(json.dumps({"registration": {"rigid_learning_rate": 1.0}}))
        assert main(["--config", str(path), "server"]) == 1


class TestRunInputErrors:
    def test_estimates_without_alignments_exits_1(
        self, tmp_path: Path, sections, detections, caplog
    ) -> None:
        from unittest.mock import patch

        from belljar.types import StepResult

        output_dir = tmp_path / "run_out"
        (output_dir / "estimates").mkdir(parents=True)
        estimates = output_dir / "estimates" / "estimates.json"
        estimates.write_text(json.dumps({"metadata": {}}))
        model_path = tmp_path / "model.pt"
        model_path.touch()

        with patch(
            "belljar.pipeline.estimate.EstimateStep.run",
            return_value=StepResult(success=True, output_path=str(estimates)),
        ):
            code = main(
                [
                    "run",
                    "--input-dir", str(sections),
                    "--output-dir", str(output_dir),
                    "--model-path", str(model_path),
                    "--detections-dir", str(detections),
                ]
            )
        assert code == 1
        assert any("estimates.json" in record.getMessage() for record in caplog.records)
