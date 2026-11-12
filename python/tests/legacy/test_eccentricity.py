"""Shape screening must keep round cells and drop elongated blobs, in v1 and v2."""

import cv2
import find_neurons
import numpy as np
import pytest

from belljar.detection import screening

IMPLEMENTATIONS = [
    pytest.param(find_neurons, id="v1"),
    pytest.param(screening, id="v2"),
]

CIRCLE_BOX = [40, 40, 80, 80]
ELLIPSE_BOX = [120, 45, 240, 75]
EDGE_BOX = [0, 0, 30, 30]


@pytest.fixture
def shapes_image():
    """Dark BGR image with a filled circle, a 3:1 filled ellipse, and a circle at the corner."""
    image = np.zeros((160, 300, 3), dtype=np.uint8)
    cv2.circle(image, (60, 60), 18, (255, 255, 255), -1)
    cv2.ellipse(image, (180, 60), (54, 12), 0, 0, 360, (255, 255, 255), -1)
    cv2.circle(image, (15, 15), 12, (255, 255, 255), -1)
    return image


@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_round_cell_kept_and_elongated_blob_removed(impl, shapes_image):
    assert impl.check_eccentricity(CIRCLE_BOX, 0.9, shapes_image) is True
    assert impl.check_eccentricity(ELLIPSE_BOX, 0.9, shapes_image) is False


@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_box_at_image_edge_is_clamped_not_wrapped(impl, shapes_image):
    assert impl.check_eccentricity(EDGE_BOX, 0.9, shapes_image) is True


@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_failed_check_keeps_box_and_warns(impl, shapes_image, monkeypatch, caplog, capsys):
    def boom(_image):
        raise RuntimeError("otsu exploded")

    monkeypatch.setattr(impl, "threshold_otsu", boom)

    with caplog.at_level("WARNING"):
        assert impl.check_eccentricity(CIRCLE_BOX, 0.9, shapes_image) is True

    output = caplog.text + capsys.readouterr().out + capsys.readouterr().err
    assert "otsu exploded" in output
    assert str(CIRCLE_BOX[0]) in output


@pytest.mark.parametrize("impl", IMPLEMENTATIONS)
def test_filter_version_is_two(impl):
    assert impl.ECCENTRICITY_FILTER_VERSION == 2


def test_v1_detection_result_records_filter_version_with_default_for_old_pickles():
    result = find_neurons.DetectionResult(boxes=[], scores=[], image_dimensions=(1, 1))
    assert result.eccentricity_filter_version == find_neurons.ECCENTRICITY_FILTER_VERSION

    legacy = find_neurons.DetectionResult.__new__(find_neurons.DetectionResult)
    legacy.__dict__.update(boxes=[], scores=[], image_dimensions=(1, 1))
    assert legacy.eccentricity_filter_version is None


def test_v1_parser_defaults_eccentricity_to_095_as_float():
    parser = find_neurons.build_parser()
    assert parser.parse_args([]).eccentricity == 0.95
    assert parser.parse_args(["-e", "0.9"]).eccentricity == 0.9
