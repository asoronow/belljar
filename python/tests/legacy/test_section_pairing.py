"""Tests for pairing prediction and annotation files by section stem."""

import pytest
from section_pairing import UnpairedSectionsError, pair_sections, section_stem


def _make_files(tmp_path, prefix, stems, suffix):
    directory = tmp_path / (prefix.rstrip("_").lower() or "images")
    directory.mkdir()
    for stem in stems:
        (directory / f"{prefix}{stem}{suffix}").write_bytes(b"")
    return directory


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("Predictions_foo.pkl", "foo"),
        ("Annotation_foo.pkl", "foo"),
        ("Annotation_foo.npy", "foo"),
        ("foo.tif", "foo"),
        ("Predictions_s01_c2.pkl", "s01_c2"),
        ("Annotation_foo.ome.pkl", "foo"),
        ("sec.1.tif", "sec"),
        ("A.TIF", "A"),
    ],
)
def test_section_stem_strips_prefix_and_extension(filename, expected):
    assert section_stem(filename) == expected


def test_pair_sections_returns_matched_pairs_in_stem_order(tmp_path):
    stems = ["s05", "s01", "s03", "s02", "s04"]
    pred_dir = _make_files(tmp_path, "Predictions_", stems, ".pkl")
    ann_dir = _make_files(tmp_path, "Annotation_", stems, ".pkl")

    pairs = pair_sections(pred_dir, ann_dir, first_suffixes=(".pkl",), second_suffixes=(".pkl",))

    assert [stem for stem, _, _ in pairs] == ["s01", "s02", "s03", "s04", "s05"]
    assert [p.name for _, p, _ in pairs] == [f"Predictions_s0{i}.pkl" for i in range(1, 6)]
    assert [a.name for _, _, a in pairs] == [f"Annotation_s0{i}.pkl" for i in range(1, 6)]


def test_pair_sections_rejects_orphan_prediction(tmp_path):
    stems = ["s01", "s02", "s03", "s04", "s05"]
    pred_dir = _make_files(tmp_path, "Predictions_", stems + ["orphan"], ".pkl")
    ann_dir = _make_files(tmp_path, "Annotation_", stems, ".pkl")

    with pytest.raises(UnpairedSectionsError) as excinfo:
        pair_sections(pred_dir, ann_dir, first_suffixes=(".pkl",), second_suffixes=(".pkl",))

    assert "orphan" in str(excinfo.value)
    assert "s01" not in str(excinfo.value)


def test_pair_sections_rejects_orphan_annotation_and_lists_every_stem(tmp_path):
    pred_dir = _make_files(tmp_path, "Predictions_", ["s01", "s02", "lonely_pred"], ".pkl")
    ann_dir = _make_files(tmp_path, "Annotation_", ["s01", "s02", "lonely_ann"], ".npy")

    with pytest.raises(UnpairedSectionsError) as excinfo:
        pair_sections(pred_dir, ann_dir, first_suffixes=(".pkl",), second_suffixes=(".pkl", ".npy"))

    message = str(excinfo.value)
    assert "lonely_pred" in message
    assert "lonely_ann" in message


def test_pair_sections_matches_images_to_annotations(tmp_path):
    img_dir = _make_files(tmp_path, "", ["a.ome", "b.ome"], ".tif")
    ann_dir = _make_files(tmp_path, "Annotation_", ["a.ome", "b.ome"], ".pkl")

    pairs = pair_sections(img_dir, ann_dir, first_suffixes=(".tif",), second_suffixes=(".pkl",))

    assert [stem for stem, _, _ in pairs] == ["a", "b"]


def test_dotted_image_name_pairs_with_first_dot_prediction_and_last_dot_annotation(tmp_path):
    img_dir = _make_files(tmp_path, "", ["sec.1"], ".tif")
    pred_dir = _make_files(tmp_path, "Predictions_", ["sec"], ".pkl")
    ann_dir = _make_files(tmp_path, "Annotation_", ["sec.1"], ".pkl")

    counted = pair_sections(pred_dir, ann_dir, first_suffixes=(".pkl",), second_suffixes=(".pkl",))
    regioned = pair_sections(img_dir, ann_dir, first_suffixes=(".tif",), second_suffixes=(".pkl",))

    assert [(stem, p.name, a.name) for stem, p, a in counted] == [
        ("sec", "Predictions_sec.pkl", "Annotation_sec.1.pkl")
    ]
    assert [(stem, i.name, a.name) for stem, i, a in regioned] == [
        ("sec", "sec.1.tif", "Annotation_sec.1.pkl")
    ]


def test_duplicate_first_dot_stems_in_one_directory_raise(tmp_path):
    img_dir = _make_files(tmp_path, "", ["foo", "foo.ome"], ".tif")
    ann_dir = _make_files(tmp_path, "Annotation_", ["foo"], ".pkl")

    with pytest.raises(UnpairedSectionsError) as excinfo:
        pair_sections(img_dir, ann_dir, first_suffixes=(".tif",), second_suffixes=(".pkl",))

    assert "foo" in str(excinfo.value)


def test_suffix_matching_ignores_case(tmp_path):
    img_dir = _make_files(tmp_path, "", ["A"], ".TIF")
    ann_dir = _make_files(tmp_path, "Annotation_", ["A"], ".PKL")

    pairs = pair_sections(
        img_dir, ann_dir, first_suffixes=(".tif", ".tiff"), second_suffixes=(".pkl",)
    )

    assert [(stem, i.name, a.name) for stem, i, a in pairs] == [("A", "A.TIF", "Annotation_A.PKL")]
