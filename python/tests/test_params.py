"""Tests for typed pipeline step parameters."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError

from belljar.atlas.provider import AtlasProvider
from belljar.config import BelljarConfig, RegistrationConfig
from belljar.pipeline.align import AlignStep
from belljar.pipeline.base import PipelineStep
from belljar.pipeline.collate import CollateStep
from belljar.pipeline.count import CountStep
from belljar.pipeline.detect import DetectStep
from belljar.pipeline.estimate import EstimateStep
from belljar.pipeline.max_projection import MaxProjectionStep
from belljar.pipeline.params import (
    AlignParams,
    CollateParams,
    CountParams,
    DetectParams,
    EstimateParams,
    MaxProjectionParams,
    SharpenParams,
)
from belljar.pipeline.sharpen import SharpenStep


class TestParamsModels:
    def test_count_params_from_json_yields_paths(self) -> None:
        payload = json.dumps(
            {"predictions_dir": "/p", "annotations_dir": "/a", "output_dir": "/o"}
        )
        params = CountParams(**json.loads(payload))
        assert params.predictions_dir == Path("/p")
        assert params.annotations_dir == Path("/a")
        assert params.output_dir == Path("/o")
        assert params.include_layers is False

    def test_missing_field_names_the_field(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            CountParams(predictions_dir="/p", output_dir="/o")
        assert "annotations_dir" in str(excinfo.value)

    @pytest.mark.parametrize(
        "cls,kwargs",
        [
            (MaxProjectionParams, {"input_dir": "/i", "output_dir": "/o"}),
            (SharpenParams, {"input_dir": "/i", "output_dir": "/o"}),
            (EstimateParams, {"input_dir": "/i", "output_dir": "/o"}),
            (AlignParams, {"input_dir": "/i", "output_dir": "/o", "alignments": []}),
            (DetectParams, {"input_dir": "/i", "output_dir": "/o", "model_path": "/m"}),
            (CountParams, {"predictions_dir": "/p", "annotations_dir": "/a", "output_dir": "/o"}),
            (CollateParams, {"input_csv": "/c", "output_dir": "/o"}),
        ],
    )
    def test_extra_fields_rejected(self, cls, kwargs) -> None:
        with pytest.raises(ValidationError) as excinfo:
            cls(**kwargs, structure_map={})
        assert "structure_map" in str(excinfo.value)

    def test_align_alignments_become_slice_alignments(self) -> None:
        params = AlignParams(
            input_dir="/i",
            output_dir="/o",
            alignments=[
                {
                    "section_name": "s0",
                    "ap_position": 10,
                    "x_angle": 0.0,
                    "y_angle": 0.0,
                    "anchoring_vectors": {"ox": 1.0},
                }
            ],
        )
        assert params.alignments[0].section_name == "s0"
        assert isinstance(params.alignments[0].ap_position, float)

    @pytest.mark.parametrize(
        "step_cls,params_cls",
        [
            (MaxProjectionStep, MaxProjectionParams),
            (SharpenStep, SharpenParams),
            (EstimateStep, EstimateParams),
            (AlignStep, AlignParams),
            (DetectStep, DetectParams),
            (CountStep, CountParams),
            (CollateStep, CollateParams),
        ],
    )
    def test_each_step_declares_its_params_class(self, step_cls, params_cls) -> None:
        assert issubclass(step_cls, PipelineStep)
        assert step_cls.params_class is params_cls


class TestConfigRejectsUnknownKeys:
    def test_registration_rejects_stale_key(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            RegistrationConfig(rigid_learning_rate=1.0)
        assert "rigid_learning_rate" in str(excinfo.value)

    def test_top_level_rejects_unknown_key(self) -> None:
        with pytest.raises(ValidationError):
            BelljarConfig.model_validate({"nonsense": 1})

    def test_roundtrip_still_loads(self, tmp_path) -> None:
        config = BelljarConfig()
        config.save(tmp_path / "c.json")
        assert BelljarConfig.load(tmp_path / "c.json") == config


def _provider_with_structures(structures: dict) -> AtlasProvider:
    atlas = MagicMock()
    atlas.structures = structures
    provider = AtlasProvider("test_atlas")
    provider._atlas = atlas
    return provider


class TestProviderStructureMap:
    def test_keys_are_plain_ints(self) -> None:
        provider = _provider_with_structures(
            {
                np.uint32(7): {
                    "name": "Region",
                    "acronym": "R",
                    "rgb_triplet": [1, 2, 3],
                    "structure_id_path": [997, 7],
                }
            }
        )
        keys = list(provider.structure_map)
        assert all(type(k) is int for k in keys)
        assert provider.structure_map[7]["acronym"] == "R"
        assert provider.structure_map[0]["acronym"] == "LIW"


def _detections(tmp_path: Path) -> Path:
    import pickle

    from belljar.types import DetectionResult

    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    result = DetectionResult(
        boxes=[[2.0, 2.0, 6.0, 6.0], [30.0, 30.0, 34.0, 34.0]],
        scores=[0.9, 0.9],
        image_width=40,
        image_height=40,
    )
    with open(pred_dir / "s01.pkl", "wb") as f:
        pickle.dump([result], f)
    return pred_dir


class TestCountUsesProviderStructureMap:
    def test_counts_with_structure_map_from_provider(self, tmp_path) -> None:
        from belljar.pipeline.base import _noop_progress

        pred_dir = _detections(tmp_path)
        ann_dir = tmp_path / "ann"
        ann_dir.mkdir()
        annotation = np.zeros((40, 40), dtype=np.uint32)
        annotation[:20] = 5
        annotation[20:] = 6
        np.save(ann_dir / "s01.npy", annotation)

        provider = _provider_with_structures(
            {
                5: {"name": "Top", "acronym": "TOP", "structure_id_path": [997, 5]},
                6: {"name": "Bottom", "acronym": "BOT", "structure_id_path": [997, 6]},
            }
        )
        step = CountStep(BelljarConfig(), atlas_provider=provider)
        params = CountParams(
            predictions_dir=pred_dir, annotations_dir=ann_dir, output_dir=tmp_path / "out"
        )
        result = step.run(_noop_progress, params)

        assert result.success, result.errors
        assert result.metrics["total_detections_assigned"] == 2
        csv_text = (tmp_path / "out" / "count_results.csv").read_text()
        assert "TOP,Top" in csv_text
        assert "BOT,Bottom" in csv_text


class TestSliceAlignmentRows:
    def test_estimate_row_fields_are_typed(self) -> None:
        params = AlignParams.model_validate(
            {
                "input_dir": "/i",
                "output_dir": "/o",
                "alignments": [
                    {
                        "section_name": "s0",
                        "ap_position": 10,
                        "x_angle": 0.0,
                        "y_angle": 0.0,
                        "anchoring_vectors": {"ox": 1.0, "oy": 0.0},
                        "uncertainty": [0.1] * 9,
                    }
                ],
            }
        )
        row = params.alignments[0]
        assert row.anchoring_vectors == {"ox": 1.0, "oy": 0.0}
        assert row.uncertainty == [0.1] * 9

    def test_misspelled_key_is_rejected(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            AlignParams.model_validate(
                {
                    "input_dir": "/i",
                    "output_dir": "/o",
                    "alignments": [
                        {
                            "section_name": "s0",
                            "ap_position": 10,
                            "x_angle": 0.0,
                            "y_angle": 0.0,
                            "linkd": False,
                        }
                    ],
                }
            )
        assert "linkd" in str(excinfo.value)
