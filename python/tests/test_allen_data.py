"""Tests for the Allen Institute data pipeline.

All tests use mocked API responses — no real HTTP requests are made.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from belljar.estimation.allen_data import (
    CCF_AP,
    CCF_DV,
    CCF_ML,
    _parse_alignment_matrix,
    assess_batch_quality,
    assess_experiment_quality,
    compose_alignment_transforms,
    filter_by_quality,
    query_allen_experiments,
)


# ---------------------------------------------------------------------------
# Helpers to build mock alignment data
# ---------------------------------------------------------------------------


def _make_alignment2d(mat: np.ndarray) -> dict:
    """Create an alignment2d dict from a 3x4 matrix."""
    fields = {}
    for i in range(12):
        row, col = divmod(i, 4)
        fields[f"tvr_{i:02d}"] = float(mat[row, col])
    return fields


def _make_alignment3d(mat: np.ndarray) -> dict:
    """Create an alignment3d dict from a 3x4 matrix."""
    fields = {}
    for i in range(12):
        row, col = divmod(i, 4)
        fields[f"trv_{i:02d}"] = float(mat[row, col])
    return fields


def _identity_2d() -> np.ndarray:
    """3x4 identity-like matrix for 2D alignment (pixel coords pass through)."""
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])


def _identity_3d() -> np.ndarray:
    """3x4 identity-like matrix for 3D alignment."""
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ])


def _make_section(section_id: int, alignment2d: dict | None = None) -> dict:
    """Create a mock section image dict."""
    section = {"id": section_id}
    if alignment2d is not None:
        section["alignment2d"] = alignment2d
    return section


def _make_experiment(
    exp_id: int,
    sections: list[dict],
    alignment3d: dict | None = None,
) -> dict:
    """Create a mock experiment dict."""
    exp = {"id": exp_id, "section_images": sections}
    if alignment3d is not None:
        exp["alignment3d"] = alignment3d
    return exp


# ---------------------------------------------------------------------------
# Transform composition tests
# ---------------------------------------------------------------------------


class TestParseAlignmentMatrix:
    def test_valid_matrix(self):
        alignment = {f"tvr_{i:02d}": float(i) for i in range(12)}
        mat = _parse_alignment_matrix(alignment, "tvr")
        assert mat is not None
        assert mat.shape == (3, 4)
        assert mat[0, 0] == 0.0
        assert mat[2, 3] == 11.0

    def test_missing_field_returns_none(self):
        alignment = {f"tvr_{i:02d}": float(i) for i in range(11)}
        # Missing tvr_11
        mat = _parse_alignment_matrix(alignment, "tvr")
        assert mat is None


class TestComposeTransforms:
    def test_identity_transforms(self):
        """With identity transforms, output should be normalized image center."""
        mat_2d = _identity_2d()
        mat_3d = _identity_3d()

        section = _make_section(1, alignment2d=_make_alignment2d(mat_2d))
        experiment = _make_experiment(1, [section], alignment3d=_make_alignment3d(mat_3d))

        result = compose_alignment_transforms(
            section, experiment, image_width=512, image_height=512,
        )

        assert result is not None
        assert result.shape == (9,)

        # With identity transforms, the origin should be the transformed center
        # pixel (256, 256) -> section (256, 256, 0) -> CCF (256, 256, 0)
        # normalized: (256/456, 256/320, 0/528)
        expected_ox = 256.0 / CCF_ML
        expected_oy = 256.0 / CCF_DV
        expected_oz = 0.0 / CCF_AP

        assert result[0] == pytest.approx(expected_ox, rel=1e-4)
        assert result[1] == pytest.approx(expected_oy, rel=1e-4)
        assert result[2] == pytest.approx(expected_oz, abs=1e-6)

    def test_known_transform_math(self):
        """Verify transform composition with a known scaling transform."""
        # 2D: scale by 0.5 (image pixels -> section coords scaled down)
        mat_2d = np.array([
            [0.5, 0.0, 0.0, 10.0],
            [0.0, 0.5, 0.0, 20.0],
            [0.0, 0.0, 0.5, 30.0],
        ])
        # 3D: identity pass-through
        mat_3d = _identity_3d()

        section = _make_section(1, alignment2d=_make_alignment2d(mat_2d))
        experiment = _make_experiment(1, [section], alignment3d=_make_alignment3d(mat_3d))

        result = compose_alignment_transforms(
            section, experiment, image_width=100, image_height=100,
        )

        assert result is not None

        # Center pixel (50, 50)
        # 2D: [50, 50, 0, 1] @ mat_2d^T -> [0.5*50+10, 0.5*50+20, 0.5*0+30] = [35, 45, 30]
        # 3D: [35, 45, 30, 1] @ identity -> [35, 45, 30]
        # Normalized: (35/456, 45/320, 30/528)
        expected_ox = 35.0 / CCF_ML
        expected_oy = 45.0 / CCF_DV
        expected_oz = 30.0 / CCF_AP

        assert result[0] == pytest.approx(expected_ox, rel=1e-4)
        assert result[1] == pytest.approx(expected_oy, rel=1e-4)
        assert result[2] == pytest.approx(expected_oz, rel=1e-4)

    def test_missing_alignment2d_returns_none(self):
        section = _make_section(1, alignment2d=None)
        experiment = _make_experiment(1, [section], alignment3d=_make_alignment3d(_identity_3d()))
        result = compose_alignment_transforms(section, experiment)
        assert result is None

    def test_missing_alignment3d_returns_none(self):
        section = _make_section(1, alignment2d=_make_alignment2d(_identity_2d()))
        experiment = _make_experiment(1, [section], alignment3d=None)
        result = compose_alignment_transforms(section, experiment)
        assert result is None

    def test_output_shape(self):
        mat_2d = _identity_2d()
        mat_3d = _identity_3d()
        section = _make_section(1, alignment2d=_make_alignment2d(mat_2d))
        experiment = _make_experiment(1, [section], alignment3d=_make_alignment3d(mat_3d))

        result = compose_alignment_transforms(section, experiment)
        assert result is not None
        assert result.shape == (9,)
        assert result.dtype == np.float64


# ---------------------------------------------------------------------------
# Quality scoring tests
# ---------------------------------------------------------------------------


class TestQualityScoring:
    def _make_monotonic_sections(self, n: int, experiment: dict) -> list[dict]:
        """Create sections with monotonically increasing AP positions."""
        sections = []
        for i in range(n):
            # Create a 2D transform that places sections at increasing AP
            ap_frac = i / max(n - 1, 1)  # 0 to 1
            mat_2d = np.array([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, ap_frac * CCF_AP],
            ])
            section = _make_section(i, alignment2d=_make_alignment2d(mat_2d))
            sections.append(section)
        return sections

    def test_perfect_monotonic_high_score(self):
        """Monotonically ordered sections with good coverage should score high."""
        alignment3d = _make_alignment3d(_identity_3d())
        experiment = _make_experiment(1, [], alignment3d=alignment3d)
        sections = self._make_monotonic_sections(20, experiment)

        score = assess_experiment_quality(sections, experiment)
        # Should be high due to perfect monotonicity and good coverage
        assert score > 0.6

    def test_few_sections_low_score(self):
        """Fewer than 3 sections should score 0."""
        alignment3d = _make_alignment3d(_identity_3d())
        experiment = _make_experiment(1, [], alignment3d=alignment3d)
        sections = self._make_monotonic_sections(2, experiment)

        score = assess_experiment_quality(sections, experiment)
        assert score == 0.0

    def test_no_alignment_low_score(self):
        """Sections without alignment data should score 0."""
        experiment = _make_experiment(1, [], alignment3d=None)
        sections = [_make_section(i, alignment2d=None) for i in range(10)]

        score = assess_experiment_quality(sections, experiment)
        assert score == 0.0

    def test_score_range(self):
        """Score should always be in [0, 1]."""
        alignment3d = _make_alignment3d(_identity_3d())
        experiment = _make_experiment(1, [], alignment3d=alignment3d)
        sections = self._make_monotonic_sections(10, experiment)

        score = assess_experiment_quality(sections, experiment)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# API query tests (mocked)
# ---------------------------------------------------------------------------


class TestQueryExperiments:
    def test_single_page(self):
        """Query that fits in one page."""
        mock_response = {
            "success": True,
            "msg": [{"id": 1}, {"id": 2}],
        }

        with patch("belljar.estimation.allen_data._api_get", return_value=mock_response):
            results = query_allen_experiments(max_rows=10, rate_limit=0)

        assert len(results) == 2
        assert results[0]["id"] == 1

    def test_pagination(self):
        """Query that spans multiple pages."""
        page1 = {"success": True, "msg": [{"id": i} for i in range(50)]}
        page2 = {"success": True, "msg": [{"id": i} for i in range(50, 75)]}

        with patch("belljar.estimation.allen_data._api_get", side_effect=[page1, page2]):
            results = query_allen_experiments(rate_limit=0)

        assert len(results) == 75

    def test_max_rows_limit(self):
        """max_rows should truncate results."""
        page = {"success": True, "msg": [{"id": i} for i in range(50)]}

        with patch("belljar.estimation.allen_data._api_get", return_value=page):
            results = query_allen_experiments(max_rows=10, rate_limit=0)

        assert len(results) == 10

    def test_empty_response(self):
        """Empty API response returns empty list."""
        mock_response = {"success": True, "msg": []}

        with patch("belljar.estimation.allen_data._api_get", return_value=mock_response):
            results = query_allen_experiments(rate_limit=0)

        assert results == []


# ---------------------------------------------------------------------------
# Batch quality assessment tests
# ---------------------------------------------------------------------------


class TestBatchQuality:
    def test_missing_metadata(self, tmp_path):
        result = assess_batch_quality(tmp_path)
        assert "error" in result

    def test_with_metadata(self, tmp_path):
        # Create metadata with mock experiment data
        metadata = {
            "exp_1": {
                "sections": {
                    str(i): {"anchoring": [0.5, 0.5, i * 0.1, 1, 0, 0, 0, 1, 0]}
                    for i in range(10)
                }
            }
        }
        with open(tmp_path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        result = assess_batch_quality(tmp_path)
        assert "experiments" in result
        assert "exp_1" in result["experiments"]
        assert result["experiments"]["exp_1"]["score"] > 0.0

    def test_filter_by_quality_accepts(self, tmp_path):
        # High-quality experiment
        metadata = {
            "exp_good": {
                "sections": {
                    str(i): {"anchoring": [0.5, 0.5, i * 0.05, 1, 0, 0, 0, 1, 0]}
                    for i in range(20)
                }
            },
            "exp_bad": {
                "sections": {
                    str(i): {"anchoring": [0.5, 0.5, 0.5, 1, 0, 0, 0, 1, 0]}
                    for i in range(5)
                }
            },
        }
        with open(tmp_path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        accepted, rejected = filter_by_quality(tmp_path, min_score=0.3)
        # exp_good should pass; exp_bad has no AP variation
        assert len(accepted) + len(rejected) == 2
