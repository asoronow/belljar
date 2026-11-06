"""File lists in the v1 scripts must order s1, s2, s10 by their numbers, not by text."""

from section_pairing import natural_key, pair_sections


def _make_files(tmp_path, prefix, stems, suffix):
    directory = tmp_path / (prefix.rstrip("_").lower() or "images")
    directory.mkdir()
    for stem in stems:
        (directory / f"{prefix}{stem}{suffix}").write_bytes(b"")
    return directory


def test_natural_key_orders_embedded_numbers_numerically():
    names = ["s10.tif", "s2.tif", "s1.tif", "s100.tif", "s20.tif"]

    assert sorted(names, key=natural_key) == [
        "s1.tif",
        "s2.tif",
        "s10.tif",
        "s20.tif",
        "s100.tif",
    ]


def test_natural_key_handles_multiple_number_runs_and_no_numbers():
    names = ["sec2_c10", "sec2_c9", "sec10_c1", "sec1_c1", "plain"]

    assert sorted(names, key=natural_key) == [
        "plain",
        "sec1_c1",
        "sec2_c9",
        "sec2_c10",
        "sec10_c1",
    ]


def test_natural_key_accepts_paths(tmp_path):
    paths = [tmp_path / "s10.png", tmp_path / "s2.png", tmp_path / "s1.png"]

    assert [p.name for p in sorted(paths, key=natural_key)] == ["s1.png", "s2.png", "s10.png"]


def test_pair_sections_returns_pairs_in_natural_stem_order(tmp_path):
    stems = ["s10", "s2", "s1", "s3"]
    pred_dir = _make_files(tmp_path, "Predictions_", stems, ".pkl")
    ann_dir = _make_files(tmp_path, "Annotation_", stems, ".pkl")

    pairs = pair_sections(pred_dir, ann_dir, first_suffixes=(".pkl",), second_suffixes=(".pkl",))

    assert [stem for stem, _, _ in pairs] == ["s1", "s2", "s3", "s10"]


def test_natural_key_ignores_non_ascii_digits():
    names = ["s²", "s1", "s10"]

    assert sorted(names, key=natural_key) == ["s1", "s10", "s²"]


def test_natural_key_is_deterministic_for_leading_zeros():
    assert sorted(["s01", "s1"], key=natural_key) == sorted(["s1", "s01"], key=natural_key)
    assert sorted(["s01", "s1"], key=natural_key) == ["s01", "s1"]
