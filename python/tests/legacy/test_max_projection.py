"""max.py must project over the z axis, not the smallest axis, and keep the image dtype."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile as tiff

max_script = pytest.importorskip("max")

MAX_PY = Path(max_script.__file__)


def _write(path, array, **kwargs):
    tiff.imwrite(path, array, **kwargs)
    return str(path)


def _project(tmp_path, name, array, **kwargs):
    src = tmp_path / "in"
    out = tmp_path / "out"
    src.mkdir()
    out.mkdir()
    max_script.process_file(_write(src / name, array, **kwargs), str(out))
    return tiff.imread(out / f"{name.split('.')[0]}.tif")


def test_z_stack_projects_over_z_keeping_uint16(tmp_path):
    stack = np.zeros((12, 512, 512), dtype=np.uint16)
    for z in range(12):
        stack[z, 10 * z, 20 * z] = 1000 + z

    projected = _project(tmp_path, "stack.tif", stack)

    assert projected.shape == (512, 512)
    assert projected.dtype == np.uint16
    for z in range(12):
        assert projected[10 * z, 20 * z] == 1000 + z
    assert projected.sum() == sum(1000 + z for z in range(12))


def test_two_dimensional_image_passes_through_unchanged(tmp_path):
    img = np.random.default_rng(0).integers(0, 65535, (512, 512), dtype=np.uint16)

    projected = _project(tmp_path, "flat.tif", img)

    assert projected.shape == (512, 512)
    assert projected.dtype == np.uint16
    np.testing.assert_array_equal(projected, img)


def test_rgb_samples_collapse_to_a_single_channel(tmp_path):
    rgb = np.zeros((512, 512, 3), dtype=np.uint8)
    rgb[..., 0] = 10
    rgb[..., 1] = 200
    rgb[..., 2] = 30

    projected = _project(tmp_path, "rgb.tif", rgb, photometric="rgb")

    assert projected.shape == (512, 512)
    assert projected.dtype == np.uint8
    assert projected.min() == 200 and projected.max() == 200


def test_zcyx_stack_projects_over_z_and_c(tmp_path):
    stack = np.zeros((3, 2, 64, 64), dtype=np.uint16)
    stack[2, 1, 5, 7] = 900
    stack[0, 0, 9, 9] = 300

    projected = _project(tmp_path, "multi.tif", stack, metadata={"axes": "ZCYX"})

    assert projected.shape == (64, 64)
    assert projected[5, 7] == 900
    assert projected[9, 9] == 300


def test_three_dimensional_array_without_axes_metadata_projects_over_axis_zero(tmp_path):
    stack = np.zeros((4, 64, 64), dtype=np.uint16)
    for z in range(4):
        stack[z, z, z] = 100 + z

    projected = _project(tmp_path, "plain.tif", stack, metadata=None, photometric="minisblack")

    assert projected.shape == (64, 64)
    for z in range(4):
        assert projected[z, z] == 100 + z


def test_ambiguous_three_dimensional_array_fails_naming_the_file_and_shape(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    (tmp_path / "out").mkdir()
    path = _write(
        src / "wide.tif",
        np.zeros((64, 64, 4), dtype=np.uint8),
        photometric="minisblack",
        metadata=None,
    )

    with pytest.raises(SystemExit) as excinfo:
        max_script.process_file(path, str(tmp_path / "out"))

    assert "wide.tif" in str(excinfo.value)
    assert "(64, 64, 4)" in str(excinfo.value)


def test_output_name_is_the_first_dot_stem(tmp_path):
    projected = _project(tmp_path, "sec.1.ome.tif", np.ones((2, 8, 8), dtype=np.uint8))

    assert projected.shape == (8, 8)
    assert (tmp_path / "out" / "sec.tif").exists()


def test_script_discovers_uppercase_tiff_extensions(tmp_path):
    src = tmp_path / "in"
    out = tmp_path / "out"
    src.mkdir()
    out.mkdir()
    _write(src / "2704_TH_10x.TIFF", np.ones((2, 8, 8), dtype=np.uint8))
    _write(src / "second.TIF", np.ones((2, 8, 8), dtype=np.uint8))
    (src / "notes.txt").write_text("skip me")

    result = subprocess.run(
        [sys.executable, str(MAX_PY), "-i", str(src), "-o", str(out), "-g", "False"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[0] == "2"
    assert sorted(p.name for p in out.iterdir()) == ["2704_TH_10x.tif", "second.tif"]


def test_axes_outside_the_supported_set_fail_naming_the_axes(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    (tmp_path / "out").mkdir()
    path = _write(
        src / "timelapse.ome.tif",
        np.zeros((2, 3, 8, 8), dtype=np.uint16),
        ome=True,
        metadata={"axes": "TZYX"},
    )

    with pytest.raises(SystemExit) as excinfo:
        max_script.process_file(path, str(tmp_path / "out"))

    assert "timelapse.ome.tif" in str(excinfo.value)
    assert "unsupported axes 'TZYX'" in str(excinfo.value)
    assert "no axis metadata" not in str(excinfo.value)


def test_script_fails_before_processing_when_output_stems_collide(tmp_path):
    src = tmp_path / "in"
    out = tmp_path / "out"
    src.mkdir()
    out.mkdir()
    _write(src / "a.1.tif", np.ones((2, 8, 8), dtype=np.uint8))
    _write(src / "a.2.tif", np.ones((2, 8, 8), dtype=np.uint8))
    _write(src / "b.tif", np.ones((2, 8, 8), dtype=np.uint8))

    result = subprocess.run(
        [sys.executable, str(MAX_PY), "-i", str(src), "-o", str(out), "-g", "False"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "a.1.tif" in result.stderr and "a.2.tif" in result.stderr
    assert list(out.iterdir()) == []
