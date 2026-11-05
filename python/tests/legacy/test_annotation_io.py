"""The v1 tools read annotations written as either .npy or .pkl."""

import pickle

import numpy as np
from section_pairing import (
    ANNOTATION_SUFFIXES,
    load_annotation,
    pair_sections,
    save_annotation,
)


def test_pairs_and_loads_npy_and_pkl_annotations(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    label = np.arange(12, dtype=np.uint32).reshape(3, 4)

    (images / "a.png").write_bytes(b"")
    (images / "b.png").write_bytes(b"")
    np.save(annotations / "Annotation_a.npy", label)
    with open(annotations / "Annotation_b.pkl", "wb") as f:
        pickle.dump(label, f)

    pairs = pair_sections(
        images, annotations, first_suffixes=(".png",), second_suffixes=ANNOTATION_SUFFIXES
    )

    assert [stem for stem, _, _ in pairs] == ["a", "b"]
    for _, _, annotation_file in pairs:
        loaded = load_annotation(annotation_file)
        assert loaded.dtype == label.dtype
        assert loaded.shape == label.shape
        np.testing.assert_array_equal(loaded, label)


def test_save_annotation_keeps_each_file_format(tmp_path):

    label = np.full((2, 2), 7, dtype=np.uint32)
    for name in ("Annotation_a.npy", "Annotation_b.pkl"):
        save_annotation(tmp_path / name, label)

    assert np.load(tmp_path / "Annotation_a.npy").dtype == np.uint32
    with open(tmp_path / "Annotation_b.pkl", "rb") as f:
        np.testing.assert_array_equal(pickle.load(f), label)
    np.testing.assert_array_equal(load_annotation(tmp_path / "Annotation_a.npy"), label)
