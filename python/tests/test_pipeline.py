"""Tests for pipeline step modules."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile as tiff

from belljar.config import BelljarConfig
from belljar.io.formats import check_unique_stems, list_images, list_tiffs
from belljar.pipeline.base import _noop_progress
from belljar.pipeline.collate import collate_counts
from belljar.pipeline.count import compute_overlaps, iou, percent_colocalized
from belljar.pipeline.max_projection import MaxProjectionStep, max_project_file
from belljar.pipeline.params import MaxProjectionParams, SharpenParams
from belljar.pipeline.sharpen import SharpenStep, enhance_contrast, sharpen_image


@pytest.fixture
def config() -> BelljarConfig:
    return BelljarConfig()


@pytest.fixture
def temp_dirs():
    """Create temporary input and output directories."""
    with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
        yield Path(input_dir), Path(output_dir)


class TestMaxProjectionStep:
    def test_validates_nonexistent_dir(self, config: BelljarConfig) -> None:
        step = MaxProjectionStep(config)
        params = MaxProjectionParams(input_dir="/nonexistent", output_dir="/tmp/out")
        errors = step.validate_inputs(params)
        assert any("does not exist" in e for e in errors)

    def test_validates_existing_dir(
        self, config: BelljarConfig, temp_dirs: tuple[Path, Path]
    ) -> None:
        input_dir, output_dir = temp_dirs
        step = MaxProjectionStep(config)
        params = MaxProjectionParams(input_dir=input_dir, output_dir=output_dir)
        assert step.validate_inputs(params) == []

    def test_no_files_returns_failure(
        self, config: BelljarConfig, temp_dirs: tuple[Path, Path]
    ) -> None:
        input_dir, output_dir = temp_dirs
        step = MaxProjectionStep(config)
        params = MaxProjectionParams(input_dir=input_dir, output_dir=output_dir)
        result = step.run(_noop_progress, params)
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
        params = MaxProjectionParams(input_dir=input_dir, output_dir=output_dir)
        result = step.run(_noop_progress, params)
        assert result.success
        assert result.metrics["files_processed"] == 1
        assert (output_dir / "test.tif").exists()


def _project(tmp_path: Path, name: str, array: np.ndarray, **kwargs) -> np.ndarray:
    src = tmp_path / name
    tiff.imwrite(src, array, **kwargs)
    return max_project_file(src)


class TestMaxProjectFile:
    def test_z_stack_projects_over_z_keeping_uint16(self, tmp_path: Path) -> None:
        stack = np.zeros((12, 512, 512), dtype=np.uint16)
        for z in range(12):
            stack[z, 10 * z, 20 * z] = 1000 + z

        projected = _project(tmp_path, "stack.tif", stack)

        assert projected.shape == (512, 512)
        assert projected.dtype == np.uint16
        for z in range(12):
            assert projected[10 * z, 20 * z] == 1000 + z
        assert projected.sum() == sum(1000 + z for z in range(12))

    def test_two_dimensional_image_passes_through(self, tmp_path: Path) -> None:
        img = np.random.default_rng(0).integers(0, 65535, (512, 512), dtype=np.uint16)

        projected = _project(tmp_path, "flat.tif", img)

        np.testing.assert_array_equal(projected, img)

    def test_rgb_samples_collapse_to_one_channel(self, tmp_path: Path) -> None:
        rgb = np.zeros((512, 512, 3), dtype=np.uint8)
        rgb[..., 1] = 200

        projected = _project(tmp_path, "rgb.tif", rgb, photometric="rgb")

        assert projected.shape == (512, 512)
        assert projected.dtype == np.uint8
        assert projected.min() == 200

    def test_zcyx_stack_projects_over_z_and_c(self, tmp_path: Path) -> None:
        stack = np.zeros((3, 2, 64, 64), dtype=np.uint16)
        stack[2, 1, 5, 7] = 900

        projected = _project(tmp_path, "multi.tif", stack, metadata={"axes": "ZCYX"})

        assert projected.shape == (64, 64)
        assert projected[5, 7] == 900

    def test_three_dimensional_without_metadata_projects_over_axis_zero(
        self, tmp_path: Path
    ) -> None:
        stack = np.zeros((4, 64, 64), dtype=np.uint16)
        for z in range(4):
            stack[z, z, z] = 100 + z

        projected = _project(tmp_path, "plain.tif", stack, metadata=None, photometric="minisblack")

        assert projected.shape == (64, 64)
        for z in range(4):
            assert projected[z, z] == 100 + z

    def test_ambiguous_three_dimensional_array_fails_naming_file_and_shape(
        self, tmp_path: Path
    ) -> None:
        src = tmp_path / "wide.tif"
        tiff.imwrite(
            src, np.zeros((64, 64, 4), dtype=np.uint8), photometric="minisblack", metadata=None
        )

        with pytest.raises(ValueError, match=r"wide\.tif.*\(64, 64, 4\)"):
            max_project_file(src)


    def test_axes_outside_the_supported_set_fail_naming_the_axes(self, tmp_path: Path) -> None:
        src = tmp_path / "timelapse.ome.tif"
        tiff.imwrite(
            src, np.zeros((2, 3, 8, 8), dtype=np.uint16), ome=True, metadata={"axes": "TZYX"}
        )

        with pytest.raises(ValueError, match=r"timelapse\.ome\.tif.*unsupported axes 'TZYX'"):
            max_project_file(src)


class TestUniqueStems:
    def test_names_every_colliding_input(self, tmp_path: Path) -> None:
        paths = [tmp_path / n for n in ["a.1.tif", "b.tif", "a.2.tif", "a.tiff"]]

        with pytest.raises(ValueError) as excinfo:
            check_unique_stems(paths)

        message = str(excinfo.value)
        assert "a.1.tif" in message and "a.2.tif" in message and "a.tiff" in message
        assert "b.tif" not in message

    def test_accepts_distinct_stems(self, tmp_path: Path) -> None:
        check_unique_stems([tmp_path / "a.tif", tmp_path / "b.ome.tif"])


class TestMaxProjectionDiscovery:
    def test_png_in_input_dir_is_ignored_without_warning(
        self, config: BelljarConfig, temp_dirs: tuple[Path, Path]
    ) -> None:
        input_dir, output_dir = temp_dirs
        tiff.imwrite(input_dir / "a.tif", np.ones((2, 8, 8), dtype=np.uint8))
        cv2.imwrite(str(input_dir / "notes.png"), np.zeros((8, 8), dtype=np.uint8))

        assert [f.name for f in list_tiffs(input_dir)] == ["a.tif"]

        step = MaxProjectionStep(config)
        params = MaxProjectionParams(input_dir=input_dir, output_dir=output_dir)
        result = step.run(_noop_progress, params)

        assert result.success
        assert result.warnings == []
        assert result.metrics == {"files_processed": 1, "files_total": 1}

    def test_colliding_output_stems_fail_before_processing(
        self, config: BelljarConfig, temp_dirs: tuple[Path, Path]
    ) -> None:
        input_dir, output_dir = temp_dirs
        tiff.imwrite(input_dir / "a.1.tif", np.ones((2, 8, 8), dtype=np.uint8))
        tiff.imwrite(input_dir / "a.2.tif", np.ones((2, 8, 8), dtype=np.uint8))

        step = MaxProjectionStep(config)
        params = MaxProjectionParams(input_dir=input_dir, output_dir=output_dir)
        result = step.run(_noop_progress, params)

        assert not result.success
        assert any("a.1.tif" in e and "a.2.tif" in e for e in result.errors)
        assert list(output_dir.iterdir()) == []

    def test_uppercase_tiff_input_is_discovered_and_projected(
        self, config: BelljarConfig, temp_dirs: tuple[Path, Path]
    ) -> None:
        input_dir, output_dir = temp_dirs
        tiff.imwrite(input_dir / "2704_TH_10x.TIFF", np.ones((2, 8, 8), dtype=np.uint16))

        assert [f.name for f in list_images(input_dir)] == ["2704_TH_10x.TIFF"]

        step = MaxProjectionStep(config)
        params = MaxProjectionParams(input_dir=input_dir, output_dir=output_dir)
        result = step.run(_noop_progress, params)

        assert result.success
        out = tiff.imread(output_dir / "2704_TH_10x.tif")
        assert out.shape == (8, 8)
        assert out.dtype == np.uint16


class TestSharpenStep:
    def test_validates_inputs(self, config: BelljarConfig) -> None:
        step = SharpenStep(config)
        params = SharpenParams(input_dir="/tmp/in", output_dir="/tmp/out")
        # /tmp/in doesn't exist
        assert len(step.validate_inputs(params)) > 0

    def test_sharpen_image_preserves_dtype(self) -> None:
        img = np.random.randint(0, 255, (50, 50), dtype=np.uint8)
        result = sharpen_image(img, equalize=False, radius=2, amount=1)
        assert result.dtype == np.uint8

    def test_run_writes_stem_dot_tif_for_any_extension_spelling(
        self, config: BelljarConfig, temp_dirs: tuple[Path, Path]
    ) -> None:
        input_dir, output_dir = temp_dirs
        tiff.imwrite(input_dir / "a.TIFF", np.zeros((32, 32), dtype=np.uint8))
        tiff.imwrite(input_dir / "b..tif", np.zeros((32, 32), dtype=np.uint16))

        step = SharpenStep(config)
        params = SharpenParams(input_dir=input_dir, output_dir=output_dir)
        result = step.run(_noop_progress, params)

        assert result.success
        assert sorted(p.name for p in output_dir.iterdir()) == ["a.tif", "b.tif"]
        assert tiff.imread(output_dir / "b.tif").dtype == np.uint16

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
            1: {"name": "Primary visual area", "acronym": "VISp", "id_path": "997/8/1"}
        }
        result = collate_counts(csv_path, structure_map, output_dir / "out.csv")
        assert "VISp" in result
        assert result["VISp"] == 10
