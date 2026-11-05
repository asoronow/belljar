"""Tests for nearest-neighbour annotation resizing and prediction dimension handling."""

import logging

import numpy as np
import pytest
from demons import resize_image_nearest_neighbor, resolve_image_dimensions

LABEL = np.uint32(7)


def _annotation_with_label():
    annotation = np.ones((200, 400), dtype=np.uint32)
    annotation[140:160, 340:360] = LABEL
    return annotation


def test_resize_returns_height_by_width():
    annotation = _annotation_with_label()

    resized = resize_image_nearest_neighbor(annotation, height=100, width=200)

    assert resized.shape == (100, 200)
    assert resized.dtype == annotation.dtype


def test_label_position_is_preserved_through_resize_and_back():
    annotation = _annotation_with_label()

    small = resize_image_nearest_neighbor(annotation, height=100, width=200)
    restored = resize_image_nearest_neighbor(small, height=200, width=400)

    assert small[75, 175] == LABEL
    assert restored.shape == (200, 400)
    assert restored[150, 350] == LABEL

    box = [340, 140, 360, 160]
    x1, y1, x2, y2 = box
    assert restored[int((y1 + y2) / 2), int((x1 + x2) / 2)] == LABEL


def test_resize_to_wider_than_tall_target_keeps_label_location():
    annotation = _annotation_with_label()

    wide = resize_image_nearest_neighbor(annotation, height=50, width=800)

    assert wide.shape == (50, 800)
    assert wide[37, 700] == LABEL


def test_resolve_image_dimensions_keeps_height_width_when_orientation_matches():
    assert resolve_image_dimensions((200, 400), reference_shape=(100, 200)) == (200, 400)


def test_resolve_image_dimensions_swaps_when_orientation_disagrees():
    assert resolve_image_dimensions((400, 200), reference_shape=(100, 200)) == (200, 400)


def test_resolve_image_dimensions_uses_boxes_when_no_reference():
    boxes = [[350.0, 150.0, 360.0, 160.0]]

    assert resolve_image_dimensions((400, 200), boxes=boxes) == (200, 400)
    assert resolve_image_dimensions((200, 400), boxes=boxes) == (200, 400)


def test_resolve_image_dimensions_square_is_unambiguous(caplog):
    with caplog.at_level(logging.WARNING):
        assert resolve_image_dimensions((300, 300)) == (300, 300)
    assert not caplog.records


def test_resolve_image_dimensions_warns_when_ambiguous(caplog):
    with caplog.at_level(logging.WARNING):
        assert resolve_image_dimensions((200, 400), reference_shape=(300, 300)) == (200, 400)
    assert any("ambiguous" in record.message.lower() for record in caplog.records)


def test_resize_rejects_positional_size():
    with pytest.raises(TypeError):
        resize_image_nearest_neighbor(_annotation_with_label(), (100, 200))
