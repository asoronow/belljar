"""Reloading a saved alignment must distinguish "absent" from "present but broken"."""

import pickle
from types import SimpleNamespace

import pytest
from alignment_io import ALIGNMENT_FILENAME, AlignmentLoadError, load_alignment


def _slice(name):
    return SimpleNamespace(
        section_name=name, ap_position=100, x_angle=1.0, y_angle=-2.0, region="A", mask=None
    )


def test_missing_alignment_returns_none(tmp_path):
    assert load_alignment(tmp_path) is None


def test_valid_alignment_round_trips(tmp_path):
    saved = {"s01.png": _slice("s01.png"), "s02.png": _slice("s02.png")}
    with open(tmp_path / ALIGNMENT_FILENAME, "wb") as f:
        pickle.dump(saved, f)

    loaded = load_alignment(tmp_path)

    assert set(loaded) == {"s01.png", "s02.png"}
    assert loaded["s02.png"].ap_position == 100


def test_corrupt_alignment_fails_naming_the_file(tmp_path):
    path = tmp_path / ALIGNMENT_FILENAME
    path.write_bytes(b"\x80\x04this is not a pickle")

    with pytest.raises(AlignmentLoadError) as excinfo:
        load_alignment(tmp_path)

    assert str(path) in str(excinfo.value)


def test_truncated_alignment_fails_naming_the_file(tmp_path):
    path = tmp_path / ALIGNMENT_FILENAME
    with open(path, "wb") as f:
        pickle.dump({"s01.png": _slice("s01.png")}, f)
    path.write_bytes(path.read_bytes()[:-10])

    with pytest.raises(AlignmentLoadError, match=str(path)):
        load_alignment(tmp_path)


def test_alignment_with_wrong_shape_fails_naming_the_file(tmp_path):
    path = tmp_path / ALIGNMENT_FILENAME
    with open(path, "wb") as f:
        pickle.dump(["not", "a", "dict"], f)

    with pytest.raises(AlignmentLoadError, match=str(path)):
        load_alignment(tmp_path)


def test_alignment_with_incompatible_slices_fails_naming_the_file(tmp_path):
    path = tmp_path / ALIGNMENT_FILENAME
    with open(path, "wb") as f:
        pickle.dump({"s01.png": SimpleNamespace(section_name="s01.png")}, f)

    with pytest.raises(AlignmentLoadError, match=str(path)):
        load_alignment(tmp_path)
