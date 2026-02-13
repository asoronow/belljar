"""Tests for pipeline step modules."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from belljar.config import BelljarConfig
from belljar.pipeline.base import _noop_progress
from belljar.pipeline.collate import collate_counts
from belljar.pipeline.count import iou, compute_overlaps, percent_colocalized
from belljar.pipeline.max_projection import MaxProjectionStep
from belljar.pipeline.sharpen import SharpenStep, sharpen_image, enhance_contrast


@pytest.fixture
def config() -> BelljarConfig:
    return BelljarConfig()


@pytest.fixture
def temp_dirs():
    """Create temporary input and output directories."""
    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        yield Path(input_dir), Path(output_dir)


class TestMaxProjectionStep:
    def test_validates_missing_input(self, config: BelljarConfig) -> None:
        step = MaxProjectionStep(config)
        errors = step.validate_inputs(output_dir="/tmp/out")
        assert any("input_dir" in e for e in errors)

    def test_validates_missing_output(self, config: BelljarConfig) -> None:
        step = MaxProjectionStep(config)
        errors = step.validate_inputs(input_dir="/tmp/in")
        assert any("output_dir" in e for e in errors)

    def test_validates_nonexistent_dir(self, config: BelljarConfig) -> None:
        step = MaxProjectionStep(config)
        errors = step.validate_inputs(input_dir="/nonexistent", output_dir="/tmp/out")
        assert any("does not exist" in e for e in errors)

    def test_no_files_returns_failure(
        self, config: BelljarConfig, temp_dirs: tuple[Path, Path]
    ) -> None:
        input_dir, output_dir = temp_dirs
        step = MaxProjectionStep(config)
        result = step.run(_noop_progress, input_dir=str(input_dir), output_dir=str(output_dir))
        assert not result.success
        assert result.errors

    def test_processes_2d_image(
        self, config: BelljarConfig, temp_dirs: tuple[Path, Path]
    ) -> None:
        input_dir, output_dir = temp_dirs
        # Create a simple 2D TIFF
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        cv2.imwrite(str(input_dir / "test.tif"), img)

        step = MaxProjectionStep(config)
        result = step.run(_noop_progress, input_dir=str(input_dir), output_dir=str(output_dir))
        assert result.success
        assert result.metrics["files_processed"] == 1
        assert (output_dir / "test.tif").exists()


class TestSharpenStep:
    def test_validates_inputs(self, config: BelljarConfig) -> None:
        step = SharpenStep(config)
        errors = step.validate_inputs(input_dir="/tmp/in", output_dir="/tmp/out")
        # /tmp/in doesn't exist
        assert len(errors) > 0

    def test_sharpen_image_preserves_dtype(self) -> None:
        img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        result = sharpen_image(img, equalize=False, radius=2, amount=1)
        assert result.dtype == np.uint8

    def test_enhance_contrast_preserves_shape(self) -> None:
        img = np.random.randint(50, 200, (30, 30), dtype=np.uint8)
        result = enhance_contrast(img, saturation_level=0.05)
        assert result.shape == img.shape
        assert result.dtype == img.dtype


class TestIoU:
    def test_identical_boxes(self) -> None:
        box = [10, 10, 50, 50]
        assert abs(iou(box, box) - 1.0) < 1e-6

    def test_no_overlap(self) -> None:
        box_a = [0, 0, 10, 10]
        box_b = [20, 20, 30, 30]
        assert iou(box_a, box_b) == 0.0

    def test_partial_overlap(self) -> None:
        box_a = [0, 0, 10, 10]
        box_b = [5, 5, 15, 15]
        result = iou(box_a, box_b)
        assert 0 < result < 1

    def test_compute_overlaps_shape(self) -> None:
        boxes1 = [[0, 0, 10, 10], [20, 20, 30, 30]]
        boxes2 = [[5, 5, 15, 15]]
        overlaps = compute_overlaps(boxes1, boxes2)
        assert overlaps.shape == (2, 1)

    def test_percent_colocalized_empty(self) -> None:
        assert percent_colocalized([], [[0, 0, 10, 10]]) == 0.0
        assert percent_colocalized([[0, 0, 10, 10]], []) == 0.0

    def test_percent_colocalized_full(self) -> None:
        boxes = [[0, 0, 10, 10], [20, 20, 30, 30]]
        assert percent_colocalized(boxes, boxes) == 100.0


class TestCollate:
    def test_collate_empty_csv(self, temp_dirs: tuple[Path, Path]) -> None:
        _, output_dir = temp_dirs
        csv_path = output_dir / "empty.csv"
        csv_path.write_text("")
        result = collate_counts(csv_path, {}, output_dir / "out.csv")
        assert result == {}

    def test_collate_basic(self, temp_dirs: tuple[Path, Path]) -> None:
        _, output_dir = temp_dirs
        csv_path = output_dir / "counts.csv"
        csv_path.write_text(
            "section_001\n"
            "Region Acronym,Region Name,Area (px),Channel #0\n"
            "VISp,Primary visual area,500,10\n"
            "\n"
        )
        structure_map = {
            np.uint32(1): {"name": "Primary visual area", "acronym": "VISp", "id_path": "997/8/1"}
        }
        result = collate_counts(csv_path, structure_map, output_dir / "out.csv")
        assert "VISp" in result
        assert result["VISp"] == 10
