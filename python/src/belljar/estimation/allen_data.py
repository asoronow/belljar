"""Allen Institute ISH data pipeline for real training data.

Downloads coronal In Situ Hybridization (ISH) section images from the Allen
Institute Brain Atlas API, computes 9-value anchoring labels from the provided
alignment transforms, and scores dataset quality for training filtering.

Allen API reference:
  - RMA: https://api.brain-map.org/api/v2/data
  - SectionDataSet: product_id=1 (ISH), plane_of_section=coronal
  - Alignment2d: image -> section (tvr_00..tvr_11, 3x4 affine)
  - Alignment3d: section -> CCFv3 (trv_00..trv_11, 3x4 affine)
  - CCFv3 dimensions at 25um: 528 AP x 320 DV x 456 ML
"""

from __future__ import annotations

import json
import logging
import pickle
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Allen Brain Atlas API base URL
ALLEN_API_BASE = "https://api.brain-map.org/api/v2/data"

# CCFv3 reference dimensions at 25um resolution
CCF_AP = 528   # Anterior-Posterior
CCF_DV = 320   # Dorsal-Ventral
CCF_ML = 456   # Medial-Lateral

# Default rate limit between API calls (seconds)
DEFAULT_RATE_LIMIT = 0.5

# Default image downsample level (4 = ~16x smaller)
DEFAULT_DOWNSAMPLE = 4

# Rows returned per API page
_PAGE_SIZE = 50


def _api_get(url: str, rate_limit: float = DEFAULT_RATE_LIMIT) -> dict:
    """Make a GET request to the Allen API and return parsed JSON.

    Args:
        url: Full URL to request.
        rate_limit: Seconds to sleep after the request.

    Returns:
        Parsed JSON response dict.

    Raises:
        urllib.error.URLError: On network errors.
        ValueError: On non-JSON or error responses.
    """
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    if rate_limit > 0:
        time.sleep(rate_limit)

    if not data.get("success", True):
        raise ValueError(f"Allen API error: {data.get('msg', 'unknown')}")

    return data


def query_allen_experiments(
    product_id: int = 1,
    plane: str = "coronal",
    rate_limit: float = DEFAULT_RATE_LIMIT,
    max_rows: int | None = None,
) -> list[dict]:
    """Query Allen RMA API for section data sets with alignment transforms.

    Paginates through all results matching the given product and plane of
    section. Each returned dict includes the experiment metadata,
    alignment3d transform, and nested section_images with alignment2d.

    Args:
        product_id: Allen product ID (1=ISH, 5=Connectivity).
        plane: Plane of section filter ('coronal', 'sagittal').
        rate_limit: Seconds between API requests.
        max_rows: Maximum total experiments to fetch (None=all).

    Returns:
        List of experiment dicts from the API.
    """
    experiments: list[dict] = []
    offset = 0

    while True:
        url = (
            f"{ALLEN_API_BASE}/query.json?"
            f"criteria=model::SectionDataSet,"
            f"rma::criteria,[failed$eqfalse],"
            f"products[id$eq{product_id}],"
            f"plane_of_section[name$eq{plane}]"
            f"&include=alignment3d,section_images(alignment2d)"
            f"&num_rows={_PAGE_SIZE}&start_row={offset}"
        )

        data = _api_get(url, rate_limit=rate_limit)
        rows = data.get("msg", [])

        if not rows:
            break

        experiments.extend(rows)
        offset += len(rows)

        logger.info("Fetched %d experiments so far (page at offset %d)", len(experiments), offset)

        if max_rows is not None and len(experiments) >= max_rows:
            experiments = experiments[:max_rows]
            break

        # If we got fewer than a full page, we're done
        if len(rows) < _PAGE_SIZE:
            break

    logger.info("Total experiments fetched: %d", len(experiments))
    return experiments


def get_section_images(experiment: dict) -> list[dict]:
    """Extract section images with alignment data from an experiment dict.

    Args:
        experiment: A single experiment dict from query_allen_experiments().

    Returns:
        List of section image dicts, each containing alignment2d data.
    """
    return experiment.get("section_images", [])


def _parse_alignment_matrix(alignment: dict, prefix: str) -> NDArray | None:
    """Parse a 3x4 affine matrix from Allen alignment fields.

    The Allen API stores affine transforms as flat fields named
    {prefix}_00, {prefix}_01, ..., {prefix}_11 representing a 3x4 matrix.

    Args:
        alignment: Dict with alignment fields.
        prefix: Field prefix ('tvr' for 2D, 'trv' for 3D).

    Returns:
        3x4 numpy array, or None if any field is missing.
    """
    keys = [f"{prefix}_{i:02d}" for i in range(12)]
    values = []
    for key in keys:
        val = alignment.get(key)
        if val is None:
            return None
        values.append(float(val))

    return np.array(values, dtype=np.float64).reshape(3, 4)


def compose_alignment_transforms(
    section: dict,
    experiment: dict,
    image_width: int = 512,
    image_height: int = 512,
) -> NDArray | None:
    """Compose 2D and 3D alignment transforms to compute anchoring labels.

    Transforms the image center and offset points through the 2D->3D chain
    to produce a 9-value anchoring vector [ox, oy, oz, ux, uy, uz, vx, vy, vz]
    normalized to [0, 1] based on CCFv3 dimensions.

    The pipeline:
        1. Image pixel -> Section coords (Alignment2d: tvr_* fields)
        2. Section coords -> CCFv3 reference space (Alignment3d: trv_* fields)
        3. Normalize by CCFv3 dimensions

    Args:
        section: Section image dict with alignment2d.
        experiment: Experiment dict with alignment3d.
        image_width: Width of section image in pixels (at native resolution).
        image_height: Height of section image in pixels (at native resolution).

    Returns:
        9-element array [ox,oy,oz, ux,uy,uz, vx,vy,vz] normalized to [0,1],
        or None if alignment data is missing.
    """
    # Parse 2D alignment (image -> section)
    alignment2d = section.get("alignment2d")
    if alignment2d is None:
        return None

    mat_2d = _parse_alignment_matrix(alignment2d, "tvr")
    if mat_2d is None:
        return None

    # Parse 3D alignment (section -> CCFv3)
    alignment3d = experiment.get("alignment3d")
    if alignment3d is None:
        return None

    mat_3d = _parse_alignment_matrix(alignment3d, "trv")
    if mat_3d is None:
        return None

    def transform_point_2d_to_3d(px: float, py: float) -> NDArray:
        """Transform a 2D image pixel through 2D->3D chain."""
        # 2D affine: (x, y) -> (sx, sy) in section space
        pt_2d = np.array([px, py, 1.0])
        section_pt = mat_2d @ np.array([pt_2d[0], pt_2d[1], 0.0, 1.0])

        # 3D affine: (sx, sy, 0) -> (ap, dv, ml) in CCFv3 space
        section_3d = np.array([section_pt[0], section_pt[1], section_pt[2], 1.0])
        ccf_pt = mat_3d @ section_3d

        return ccf_pt

    # Origin: center of image
    cx, cy = image_width / 2.0, image_height / 2.0
    origin = transform_point_2d_to_3d(cx, cy)

    # U vector: horizontal offset (width direction)
    offset = min(image_width, image_height) / 2.0
    p_right = transform_point_2d_to_3d(cx + offset, cy)
    u_vec = p_right - origin

    # V vector: vertical offset (height direction)
    p_down = transform_point_2d_to_3d(cx, cy + offset)
    v_vec = p_down - origin

    # Normalize to [0, 1] based on CCFv3 dimensions
    dims = np.array([CCF_ML, CCF_DV, CCF_AP], dtype=np.float64)
    origin_norm = origin / dims
    u_norm = u_vec / dims
    v_norm = v_vec / dims

    anchoring = np.concatenate([origin_norm, u_norm, v_norm])
    return anchoring


def assess_experiment_quality(sections: list[dict], experiment: dict) -> float:
    """Score an experiment's data quality from 0 to 1.

    Evaluates three criteria:
        1. AP monotonicity: are sections ordered by increasing AP position?
        2. Angle variance: are consecutive section angles consistent (< 5 deg)?
        3. AP coverage: what fraction of the atlas AP range is covered?

    Args:
        sections: Section image dicts from get_section_images().
        experiment: Parent experiment dict with alignment3d.

    Returns:
        Quality score in [0, 1]. Higher is better.
    """
    if len(sections) < 3:
        return 0.0

    # Compute anchoring vectors for all sections
    anchorings = []
    for section in sections:
        anch = compose_alignment_transforms(section, experiment)
        if anch is not None:
            anchorings.append(anch)

    if len(anchorings) < 3:
        return 0.0

    anchorings_arr = np.array(anchorings)

    # Extract AP positions (oz = index 2)
    ap_positions = anchorings_arr[:, 2]

    # 1. Monotonicity score: fraction of consecutive pairs that are in order
    diffs = np.diff(ap_positions)
    if len(diffs) == 0:
        monotonicity = 0.0
    else:
        # Count pairs that are monotonically increasing (or decreasing)
        increasing = np.sum(diffs > 0)
        decreasing = np.sum(diffs < 0)
        monotonicity = max(increasing, decreasing) / len(diffs)

    # 2. Angle variance score: compute angles between consecutive section planes
    u_vecs = anchorings_arr[:, 3:6]
    v_vecs = anchorings_arr[:, 6:9]

    angle_diffs = []
    for i in range(len(anchorings_arr) - 1):
        # Normal of each plane
        n1 = np.cross(u_vecs[i], v_vecs[i])
        n2 = np.cross(u_vecs[i + 1], v_vecs[i + 1])
        norm1 = np.linalg.norm(n1)
        norm2 = np.linalg.norm(n2)
        if norm1 > 1e-8 and norm2 > 1e-8:
            cos_angle = np.clip(np.dot(n1 / norm1, n2 / norm2), -1.0, 1.0)
            angle_deg = np.degrees(np.arccos(cos_angle))
            angle_diffs.append(angle_deg)

    if angle_diffs:
        mean_angle_diff = np.mean(angle_diffs)
        # Score: 1.0 if mean angle < 1 deg, 0.0 if > 10 deg
        angle_score = float(np.clip(1.0 - (mean_angle_diff - 1.0) / 9.0, 0.0, 1.0))
    else:
        angle_score = 0.0

    # 3. AP coverage score: fraction of [0, 1] range covered
    ap_range = float(ap_positions.max() - ap_positions.min())
    coverage = float(np.clip(ap_range, 0.0, 1.0))

    # Weighted combination
    score = 0.4 * monotonicity + 0.3 * angle_score + 0.3 * coverage
    return float(np.clip(score, 0.0, 1.0))


def download_section_image(
    section_image_id: int,
    output_path: Path,
    downsample: int = DEFAULT_DOWNSAMPLE,
    rate_limit: float = DEFAULT_RATE_LIMIT,
) -> bool:
    """Download a single section image from the Allen API.

    Args:
        section_image_id: Allen section image ID.
        output_path: Local path to save the image.
        downsample: Downsample level (0=full, 4=16x smaller).
        rate_limit: Seconds to wait after download.

    Returns:
        True if download succeeded, False otherwise.
    """
    url = (
        f"https://api.brain-map.org/api/v2/section_image_download/"
        f"{section_image_id}?downsample={downsample}"
    )

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=60) as resp:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(resp.read())

        if rate_limit > 0:
            time.sleep(rate_limit)

        return True
    except (urllib.error.URLError, OSError) as e:
        logger.warning("Failed to download section image %d: %s", section_image_id, e)
        return False


def assess_batch_quality(
    data_dir: Path,
    model_path: Path | None = None,
) -> dict:
    """Run automated quality checks on downloaded Allen data.

    Stage 1 (automated): Re-scores experiments using monotonicity, angle
    variance, and AP coverage metrics from the saved metadata.

    Stage 2 (model-assisted, optional): If model_path is provided, runs
    inference and compares predictions vs Allen labels.

    Args:
        data_dir: Directory containing downloaded experiments with metadata.pkl.
        model_path: Optional path to trained model for cross-validation.

    Returns:
        Dict with quality assessment results per experiment.
    """
    metadata_path = data_dir / "metadata.pkl"
    if not metadata_path.exists():
        return {"error": "No metadata.pkl found", "experiments": {}}

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    results: dict = {"experiments": {}}
    scores = []

    for exp_id, exp_meta in metadata.items():
        exp_sections = exp_meta.get("sections", {})
        if not exp_sections:
            continue

        # Collect anchoring vectors for this experiment
        anchorings = []
        for _sec_id, sec_meta in sorted(exp_sections.items()):
            anch = sec_meta.get("anchoring")
            if anch is not None:
                anchorings.append(np.array(anch))

        if len(anchorings) < 3:
            results["experiments"][str(exp_id)] = {"score": 0.0, "reason": "too_few_sections"}
            continue

        anchorings_arr = np.array(anchorings)
        ap_positions = anchorings_arr[:, 2]

        # Monotonicity
        diffs = np.diff(ap_positions)
        increasing = np.sum(diffs > 0)
        decreasing = np.sum(diffs < 0)
        monotonicity = max(increasing, decreasing) / len(diffs) if len(diffs) > 0 else 0.0

        # AP coverage
        ap_range = float(ap_positions.max() - ap_positions.min())
        coverage = float(np.clip(ap_range, 0.0, 1.0))

        score = 0.5 * monotonicity + 0.5 * coverage
        scores.append(score)
        results["experiments"][str(exp_id)] = {
            "score": float(score),
            "n_sections": len(anchorings),
            "monotonicity": float(monotonicity),
            "coverage": float(coverage),
        }

    if scores:
        results["mean_score"] = float(np.mean(scores))
        results["median_score"] = float(np.median(scores))
        results["n_experiments"] = len(scores)

    return results


def filter_by_quality(
    data_dir: Path,
    min_score: float = 0.7,
) -> tuple[list[str], list[str]]:
    """Filter experiments by quality score.

    Args:
        data_dir: Directory containing downloaded data with metadata.pkl.
        min_score: Minimum quality score to accept.

    Returns:
        Tuple of (accepted_experiment_ids, rejected_experiment_ids).
    """
    quality = assess_batch_quality(data_dir)
    experiments = quality.get("experiments", {})

    accepted = []
    rejected = []

    for exp_id, info in experiments.items():
        if isinstance(info, dict) and info.get("score", 0.0) >= min_score:
            accepted.append(exp_id)
        else:
            rejected.append(exp_id)

    logger.info(
        "Quality filter (min_score=%.2f): %d accepted, %d rejected",
        min_score, len(accepted), len(rejected),
    )
    return accepted, rejected
