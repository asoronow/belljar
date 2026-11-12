"""sharpen.py must write <stem>.tif and discover every tif extension spelling."""

from types import SimpleNamespace

import numpy as np
import pytest
import tifffile as tiff

sharpen = pytest.importorskip("sharpen")


def _sharpen(tmp_path, name, img):
    src = tmp_path / "in"
    out = tmp_path / "out"
    src.mkdir()
    out.mkdir()
    tiff.imwrite(src / name, img)
    args = SimpleNamespace(equalize=False)
    sharpen.process_file(src / name, out, args, radius=2, amount=1)
    return out


def test_output_is_stem_dot_tif_without_a_double_dot(tmp_path):
    img = np.random.default_rng(0).integers(0, 255, (64, 64), dtype=np.uint8)

    out = _sharpen(tmp_path, "foo.tif", img)

    assert sorted(p.name for p in out.iterdir()) == ["foo.tif"]


def test_output_stem_is_the_name_up_to_the_first_dot(tmp_path):
    out = _sharpen(tmp_path, "sec.1.ome.tiff", np.zeros((32, 32), dtype=np.uint8))

    assert sorted(p.name for p in out.iterdir()) == ["sec.tif"]


def test_uint16_input_stays_uint16(tmp_path):
    img = np.random.default_rng(1).integers(0, 65535, (32, 32), dtype=np.uint16)

    out = _sharpen(tmp_path, "deep.tif", img)

    assert tiff.imread(out / "deep.tif").dtype == np.uint16


def test_script_fails_before_processing_when_output_stems_collide(tmp_path):
    import subprocess
    import sys

    src = tmp_path / "in"
    out = tmp_path / "out"
    src.mkdir()
    out.mkdir()
    tiff.imwrite(src / "a.tif", np.zeros((8, 8), dtype=np.uint8))
    tiff.imwrite(src / "a.tiff", np.zeros((8, 8), dtype=np.uint8))

    result = subprocess.run(
        [sys.executable, sharpen.__file__, "-i", str(src), "-o", str(out)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "a.tif" in result.stderr and "a.tiff" in result.stderr
    assert list(out.iterdir()) == []
