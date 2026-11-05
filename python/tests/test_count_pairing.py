"""Tests for matching prediction files to annotation files in the count step."""

import pytest

from belljar.pipeline.count import SectionPairingError, pair_section_files


def _touch(directory, names):
    directory.mkdir()
    for name in names:
        (directory / name).write_bytes(b"")
    return directory


def test_pair_section_files_returns_pairs_in_stem_order(tmp_path):
    stems = ["s05", "s01", "s03", "s02", "s04"]
    pred_dir = _touch(tmp_path / "pred", [f"{s}.pkl" for s in stems])
    ann_dir = _touch(tmp_path / "ann", [f"{s}.npy" for s in stems])

    pairs = pair_section_files(pred_dir, ann_dir)

    assert [stem for stem, _, _ in pairs] == ["s01", "s02", "s03", "s04", "s05"]
    assert [p.name for _, p, _ in pairs] == [f"s0{i}.pkl" for i in range(1, 6)]
    assert [a.name for _, _, a in pairs] == [f"s0{i}.npy" for i in range(1, 6)]


def test_pair_section_files_accepts_legacy_prefixes(tmp_path):
    pred_dir = _touch(tmp_path / "pred", ["Predictions_s01.pkl"])
    ann_dir = _touch(tmp_path / "ann", ["Annotation_s01.pkl"])

    pairs = pair_section_files(pred_dir, ann_dir)

    assert [stem for stem, _, _ in pairs] == ["s01"]


def test_pair_section_files_rejects_orphan_and_never_pairs_by_position(tmp_path):
    stems = ["s01", "s02", "s03", "s04", "s05"]
    pred_dir = _touch(tmp_path / "pred", [f"{s}.pkl" for s in stems + ["orphan"]])
    ann_dir = _touch(tmp_path / "ann", [f"{s}.npy" for s in stems])

    with pytest.raises(SectionPairingError) as excinfo:
        pair_section_files(pred_dir, ann_dir)

    assert "orphan" in str(excinfo.value)
    assert "s01" not in str(excinfo.value)


def test_pair_section_files_lists_orphans_on_both_sides(tmp_path):
    pred_dir = _touch(tmp_path / "pred", ["s01.pkl", "only_pred.pkl"])
    ann_dir = _touch(tmp_path / "ann", ["s01.npy", "only_ann.npy"])

    with pytest.raises(SectionPairingError) as excinfo:
        pair_section_files(pred_dir, ann_dir)

    message = str(excinfo.value)
    assert "only_pred" in message
    assert "only_ann" in message


def test_pair_section_files_keys_on_first_dot_stem(tmp_path):
    pred_dir = _touch(tmp_path / "pred", ["Predictions_sec.pkl"])
    ann_dir = _touch(tmp_path / "ann", ["Annotation_sec.1.npy"])

    pairs = pair_section_files(pred_dir, ann_dir)

    assert [(stem, p.name, a.name) for stem, p, a in pairs] == [
        ("sec", "Predictions_sec.pkl", "Annotation_sec.1.npy")
    ]


def test_pair_section_files_rejects_duplicate_first_dot_stems(tmp_path):
    pred_dir = _touch(tmp_path / "pred", ["foo.pkl", "foo.ome.pkl"])
    ann_dir = _touch(tmp_path / "ann", ["foo.npy"])

    with pytest.raises(SectionPairingError) as excinfo:
        pair_section_files(pred_dir, ann_dir)

    assert "foo" in str(excinfo.value)


def test_pair_section_files_ignores_suffix_case(tmp_path):
    pred_dir = _touch(tmp_path / "pred", ["s01.PKL"])
    ann_dir = _touch(tmp_path / "ann", ["s01.NPY"])

    pairs = pair_section_files(pred_dir, ann_dir)

    assert [(stem, p.name, a.name) for stem, p, a in pairs] == [("s01", "s01.PKL", "s01.NPY")]
