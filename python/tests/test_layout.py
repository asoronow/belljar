"""Align writes the flat per-section files that count pairs by stem."""

import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest

from belljar.config import BelljarConfig
from belljar.pipeline.align import AlignStep
from belljar.pipeline.base import _noop_progress
from belljar.pipeline.count import CountStep
from belljar.pipeline.params import AlignParams, CountParams
from belljar.types import DetectionResult

STEMS = ["sec_a", "sec_b"]


@pytest.fixture
def fast_config() -> BelljarConfig:
    config = BelljarConfig()
    config.registration.processing_resolution = 128
    config.registration.rigid_iterations = 5
    config.registration.affine_iterations = 5
    config.registration.bspline_iterations = 5
    config.registration.bspline_grid_size = 3
    return config


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
            boxes=[[10.0, 10.0, 20.0, 20.0], [120.0, 120.0, 130.0, 130.0]],
            scores=[0.9, 0.9],
            image_width=256,
            image_height=256,
        )
        with open(pred_dir / f"{stem}.pkl", "wb") as f:
            pickle.dump([result], f)
    return pred_dir


def _count(config, provider, predictions_dir: Path, annotations_dir: Path, out: Path):
    step = CountStep(config, atlas_provider=provider)
    params = CountParams(
        predictions_dir=predictions_dir, annotations_dir=annotations_dir, output_dir=out
    )
    return step.run(_noop_progress, params)


class TestAlignThenCount:
    def test_align_writes_flat_files_count_can_pair(
        self, tmp_path, sections, detections, fast_config, structured_provider
    ) -> None:
        aligned = tmp_path / "aligned"
        step = AlignStep(fast_config, atlas_provider=structured_provider)
        alignments = [
            {"section_name": stem, "ap_position": 50.0, "x_angle": 0.0, "y_angle": 0.0}
            for stem in STEMS
        ]
        params = AlignParams.model_validate(
            {"input_dir": sections, "output_dir": aligned, "alignments": alignments}
        )
        result = step.run(_noop_progress, params)
        assert result.success, result.errors

        names = sorted(p.name for p in aligned.iterdir())
        assert names == sorted(
            f"{prefix}_{stem}.{ext}"
            for stem in STEMS
            for prefix, ext in [
                ("Annotation", "npy"),
                ("Atlas", "png"),
                ("Label", "png"),
                ("Composite", "png"),
            ]
        )
        labels = np.load(aligned / "Annotation_sec_a.npy")
        assert labels.dtype == np.uint32
        assert labels.shape == (256, 256)
        assert set(np.unique(labels)) <= {0, 1, 2, 3}
        assert cv2.imread(str(aligned / "Composite_sec_a.png")).shape == (256, 256, 3)

        count = _count(fast_config, structured_provider, detections, aligned, tmp_path / "out")
        assert count.success, count.errors
        assert count.metrics["sections_counted"] == len(STEMS)
        assert count.metrics["total_detections_assigned"] == 2 * len(STEMS)

    def test_legacy_pickle_annotations_still_count(
        self, tmp_path, detections, fast_config, structured_provider
    ) -> None:
        aligned = tmp_path / "legacy"
        aligned.mkdir()
        annotation = np.zeros((256, 256), dtype=np.uint32)
        annotation[:128] = 1
        annotation[128:] = 2
        for stem in STEMS:
            with open(aligned / f"Annotation_{stem}.pkl", "wb") as f:
                pickle.dump(annotation, f)

        count = _count(fast_config, structured_provider, detections, aligned, tmp_path / "out")
        assert count.success, count.errors
        assert count.metrics["total_detections_assigned"] == 2 * len(STEMS)
