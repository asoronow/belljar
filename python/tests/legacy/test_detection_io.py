"""Predictions pickles written by find_neurons.py run as __main__ must load from any script."""

import pickle
import sys
import types

import find_neurons
import pytest
from section_pairing import load_detection_result


def _pickle_as_main(obj):
    """Serialize obj with its class path rewritten to __main__.DetectionResult."""
    fake_main = types.ModuleType("__main__")
    fake_main.DetectionResult = find_neurons.DetectionResult
    saved = sys.modules["__main__"]
    saved_module = find_neurons.DetectionResult.__module__
    sys.modules["__main__"] = fake_main
    find_neurons.DetectionResult.__module__ = "__main__"
    try:
        return pickle.dumps(obj)
    finally:
        find_neurons.DetectionResult.__module__ = saved_module
        sys.modules["__main__"] = saved


@pytest.fixture
def main_pickle(tmp_path):
    result = find_neurons.DetectionResult([[1, 2, 3, 4]], [0.9], (100, 200))
    data = _pickle_as_main([result])
    assert b"__main__" in data
    path = tmp_path / "Predictions_s1.pkl"
    path.write_bytes(data)
    return path


def test_plain_pickle_load_cannot_resolve_main_class(main_pickle):
    with open(main_pickle, "rb") as f, pytest.raises(AttributeError):
        pickle.load(f)


def test_load_detection_result_maps_main_class(main_pickle):
    predictions = load_detection_result(main_pickle)

    assert len(predictions) == 1
    assert predictions[0].boxes == [[1, 2, 3, 4]]
    assert predictions[0].scores == [0.9]
    assert predictions[0].image_dimensions == (100, 200)
    assert isinstance(predictions[0], find_neurons.DetectionResult)
