"""Classical multi-stage registration using SimpleITK.

Refactored from demons.py with configurable parameters, higher default
resolution, and finer B-spline grid.

Pipeline: Rigid (Euler2D) -> Affine -> B-spline (deformable)
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
import SimpleITK as sitk
from numpy.typing import NDArray

from belljar.config import RegistrationConfig
from belljar.registration.metrics import compute_registration_quality
from belljar.registration.preprocessing import (
    apply_layer_intensity_adjustments,
    match_histograms,
    preprocess_for_registration,
    resize_nearest_neighbor,
)
from belljar.types import RegistrationMetrics

logger = logging.getLogger(__name__)


def multimodal_registration(
    fixed: sitk.Image,
    moving: sitk.Image,
    config: RegistrationConfig | None = None,
) -> sitk.CompositeTransform:
    """Run hierarchical rigid -> affine -> B-spline registration.

    Args:
        fixed: Reference image (tissue section, edge-enhanced).
        moving: Moving image (atlas slice, edge-enhanced).
        config: Registration parameters. Uses defaults if None.

    Returns:
        Composite transform (rigid + affine + B-spline).
    """
    if config is None:
        config = RegistrationConfig()

    fixed_pre = preprocess_for_registration(fixed)
    moving_pre = preprocess_for_registration(moving)

    # Stage 1: Rigid registration
    rigid_tx = sitk.CenteredTransformInitializer(
        fixed_pre,
        moving_pre,
        sitk.Euler2DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation()
    R.SetOptimizerAsGradientDescent(
        learningRate=config.rigid_learning_rate,
        numberOfIterations=config.rigid_iterations,
        convergenceMinimumValue=1e-8,
        convergenceWindowSize=20,
    )
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[3, 2, 0])
    R.SetInitialTransform(rigid_tx)
    R.SetInterpolator(sitk.sitkLinear)

    out_rigid = R.Execute(fixed_pre, moving_pre)
    logger.debug("Rigid registration complete")

    rigid_moving = sitk.Resample(
        moving_pre, fixed_pre, out_rigid, sitk.sitkLinear, 0.0, moving_pre.GetPixelID()
    )

    # Stage 2: Affine registration
    affine_tx = sitk.CenteredTransformInitializer(
        fixed_pre,
        rigid_moving,
        sitk.AffineTransform(fixed_pre.GetDimension()),
    )

    R2 = sitk.ImageRegistrationMethod()
    R2.SetMetricAsMattesMutualInformation()
    R2.SetOptimizerAsGradientDescent(
        learningRate=config.affine_learning_rate,
        numberOfIterations=config.affine_iterations,
        convergenceMinimumValue=1e-10,
        convergenceWindowSize=10,
    )
    R2.SetOptimizerScalesFromPhysicalShift()
    R2.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    R2.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    R2.SetInitialTransform(affine_tx)
    R2.SetInterpolator(sitk.sitkLinear)

    out_affine = R2.Execute(fixed_pre, rigid_moving)
    logger.debug("Affine registration complete")

    resampled_moving = sitk.Resample(
        rigid_moving, fixed_pre, out_affine, sitk.sitkLinear, 0.0, moving_pre.GetPixelID()
    )

    # Stage 3: B-spline deformable registration
    grid_size = [config.bspline_grid_size] * fixed_pre.GetDimension()
    bspline_tx = sitk.BSplineTransformInitializer(fixed_pre, grid_size)

    R3 = sitk.ImageRegistrationMethod()
    R3.SetMetricAsMattesMutualInformation()
    R3.SetOptimizerAsGradientDescent(
        learningRate=config.bspline_learning_rate,
        numberOfIterations=config.bspline_iterations,
        convergenceMinimumValue=1e-12,
        convergenceWindowSize=20,
    )
    R3.SetOptimizerScalesFromPhysicalShift()
    R3.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    R3.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    R3.SetInitialTransform(bspline_tx, inPlace=False)
    R3.SetInterpolator(sitk.sitkLinear)

    out_bspline = R3.Execute(fixed_pre, resampled_moving)
    logger.debug("B-spline registration complete")

    # Compose all transforms
    composite = sitk.CompositeTransform(fixed_pre.GetDimension())
    composite.AddTransform(out_rigid)
    composite.AddTransform(out_affine)
    composite.AddTransform(out_bspline)

    return composite


def register_to_atlas(
    tissue: NDArray,
    section: NDArray,
    label: NDArray,
    structure_map: dict,
    config: RegistrationConfig | None = None,
) -> tuple[NDArray, NDArray, NDArray, RegistrationMetrics]:
    """Register a tissue section to an atlas slice.

    Args:
        tissue: Tissue image (grayscale uint8).
        section: Atlas slice image (grayscale uint8).
        label: Annotation slice (uint32 region IDs).
        structure_map: Region ID -> metadata dict.
        config: Registration configuration.

    Returns:
        Tuple of (warped_labels, warped_atlas, color_label, metrics).
    """
    if config is None:
        config = RegistrationConfig()

    res = config.processing_resolution

    # Resize to processing resolution
    tissue_resized = cv2.resize(tissue, (res, res))
    section_resized = cv2.resize(section, (res, res))
    label_resized = resize_nearest_neighbor(label, (res, res))

    # Apply layer-specific intensity adjustments (vectorized)
    section_resized = apply_layer_intensity_adjustments(
        section_resized,
        label_resized,
        structure_map,
        config.layer_intensity_adjustments,
    )

    fixed = sitk.GetImageFromArray(tissue_resized, isVector=False)
    moving = sitk.GetImageFromArray(section_resized, isVector=False)
    label_sitk = sitk.GetImageFromArray(label_resized, isVector=False)

    # Histogram matching
    fixed = match_histograms(
        fixed, moving, config.histogram_levels, config.histogram_match_points
    )

    # Cast to float32 for registration
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)

    # Run multi-stage registration
    transform = multimodal_registration(fixed, moving, config)

    # Apply transform to label (nearest-neighbor to preserve IDs)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(0)

    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputPixelType(sitk.sitkUInt32)
    resampled_label = resampler.Execute(label_sitk)

    # Apply transform to atlas (linear interpolation)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputPixelType(sitk.sitkUInt8)
    resampled_atlas = resampler.Execute(moving)

    # Compute quality metrics
    metrics = compute_registration_quality(fixed, sitk.Cast(resampled_atlas, sitk.sitkFloat32))

    # Build color label image
    label_array = sitk.GetArrayFromImage(resampled_label)
    color_label = np.zeros((*label_array.shape, 3), dtype=np.uint8)
    for region_id, info in structure_map.items():
        mask = label_array == region_id
        if np.any(mask):
            color_label[mask] = info["color"]

    color_label = cv2.cvtColor(color_label, cv2.COLOR_RGB2BGR)

    # Convert back to numpy
    resampled_label_array = sitk.GetArrayFromImage(resampled_label)
    resampled_atlas_array = sitk.GetArrayFromImage(resampled_atlas)

    # Resize back to original tissue dimensions
    original_size = tissue.shape[:2][::-1]  # (width, height)
    resampled_atlas_array = cv2.resize(resampled_atlas_array, original_size)
    color_label = cv2.resize(color_label, original_size)
    color_label = cv2.cvtColor(color_label, cv2.COLOR_BGR2RGB)
    resampled_label_array = resize_nearest_neighbor(resampled_label_array, original_size)

    return resampled_label_array, resampled_atlas_array, color_label, metrics
