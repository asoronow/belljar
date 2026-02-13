"""Registration quality metrics.

Provides quantitative measures for evaluating how well a tissue section
was registered to the atlas.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from belljar.types import RegistrationMetrics


def compute_registration_quality(
    fixed: sitk.Image,
    warped_moving: sitk.Image,
) -> RegistrationMetrics:
    """Compute quality metrics for a registration result.

    Args:
        fixed: Reference image (the tissue section).
        warped_moving: Atlas image warped to match the tissue.

    Returns:
        RegistrationMetrics with MI and NCC values.
    """
    # Mutual Information
    mi_filter = sitk.ImageRegistrationMethod()
    mi_filter.SetMetricAsMattesMutualInformation()
    mi_value = mi_filter.MetricEvaluate(fixed, warped_moving)

    # Normalized Cross-Correlation
    fixed_array = sitk.GetArrayFromImage(fixed).ravel().astype(np.float64)
    warped_array = sitk.GetArrayFromImage(warped_moving).ravel().astype(np.float64)

    if fixed_array.std() > 0 and warped_array.std() > 0:
        ncc = float(np.corrcoef(fixed_array, warped_array)[0, 1])
    else:
        ncc = 0.0

    return RegistrationMetrics(
        mutual_information=float(mi_value),
        normalized_cross_correlation=ncc,
    )
