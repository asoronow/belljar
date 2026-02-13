"""Tests for detection post-processing / screening."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from belljar.detection.screening import screen_predictions, xyxy_to_area


class MockBbox:
    """Mock for SAHI bbox object."""
    def __init__(self, xyxy: list[float]) -> None:
        self._xyxy = xyxy

    def to_xyxy(self) -> list[float]:
        return self._xyxy


class MockScore:
    """Mock for SAHI score object."""
    def __init__(self, value: float) -> None:
        self.value = value


class MockPrediction:
    """Mock for SAHI ObjectPrediction."""
    def __init__(self, xyxy: list[float], score: float = 0.9) -> None:
        self.bbox = MockBbox(xyxy)
        self.score = MockScore(score)


class TestXyxyToArea:
    def test_simple_area(self) -> None:
        assert xyxy_to_area([0, 0, 10, 10]) == 100

    def test_non_zero_origin(self) -> None:
        assert xyxy_to_area([5, 5, 15, 20]) == 150

    def test_zero_area(self) -> None:
        assert xyxy_to_area([5, 5, 5, 5]) == 0


class TestScreenPredictions:
    def test_area_filtering(self) -> None:
        """Objects below area threshold should be removed."""
        preds = [
            MockPrediction([0, 0, 5, 5]),     # area = 25 (below threshold)
            MockPrediction([0, 0, 20, 20]),    # area = 400 (above threshold)
            MockPrediction([0, 0, 30, 30]),    # area = 900 (above threshold)
            MockPrediction([0, 0, 25, 25]),    # area = 625 (above threshold)
        ]
        result = screen_predictions(preds, area_threshold=200)
        assert len(result) == 3

    def test_statistical_outlier_removal(self) -> None:
        """Objects with area > mean + 2*std should be removed."""
        # Many normal-sized detections plus one outlier
        preds = [
            MockPrediction([0, 0, 15, 15]),   # area = 225
            MockPrediction([0, 0, 16, 16]),   # area = 256
            MockPrediction([0, 0, 14, 14]),   # area = 196
            MockPrediction([0, 0, 15, 15]),   # area = 225
            MockPrediction([0, 0, 15, 14]),   # area = 210
            MockPrediction([0, 0, 16, 15]),   # area = 240
            MockPrediction([0, 0, 14, 15]),   # area = 210
            MockPrediction([0, 0, 200, 200]), # area = 40000 (clear outlier)
        ]
        result = screen_predictions(preds, area_threshold=100)
        # With 7 normal objects ~220 area and 1 outlier at 40000,
        # the outlier should be filtered by mean + 2*std
        assert len(result) < len(preds)

    def test_empty_input(self) -> None:
        """Empty input should return empty output."""
        result = screen_predictions([], area_threshold=100)
        assert result == []

    def test_few_predictions_skip_statistical(self) -> None:
        """With < 3 predictions after area filter, skip statistical filtering."""
        preds = [
            MockPrediction([0, 0, 20, 20]),  # area = 400
            MockPrediction([0, 0, 5, 5]),    # area = 25 (filtered)
        ]
        result = screen_predictions(preds, area_threshold=100)
        assert len(result) == 1

    def test_eccentricity_skipped_without_image(self) -> None:
        """Eccentricity filtering should be skipped when no image is provided."""
        preds = [
            MockPrediction([0, 0, 20, 20]),
            MockPrediction([0, 0, 25, 25]),
            MockPrediction([0, 0, 22, 22]),
        ]
        # Should not raise, just warn
        result = screen_predictions(
            preds, area_threshold=100, eccentricity_threshold=0.5, image=None
        )
        assert len(result) == 3
