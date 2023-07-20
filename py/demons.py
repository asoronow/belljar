import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import nrrd
from PIL import Image
import pickle


def b_spline_registration(fixed, moving):
    """Register two images using a B-Spline transformation. Returns the displacement field."""
    # Create a B-Spline transform
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(8)

    transform_domain_mesh_size = [2] * fixed.GetDimension()

    tx = sitk.BSplineTransformInitializer(
        fixed,
        transform_domain_mesh_size,
    )

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsJointHistogramMutualInformation()

    # Use GD to optimize the BSpline coefficients.
    R.SetOptimizerAsGradientDescentLineSearch(
        5.0,
        200,
        convergenceMinimumValue=1e-7,
        convergenceWindowSize=5,
    )

    R.SetInterpolator(sitk.sitkLinear)

    # Initialize registration with identity transform
    R.SetInitialTransformAsBSpline(tx, inPlace=True, scaleFactors=[1, 2, 5])

    # Shrink levels for faster computation
    R.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[4, 2, 1])

    out = R.Execute(fixed, moving)

    filter = sitk.TransformToDisplacementFieldFilter()
    filter.SetReferenceImage(fixed)
    displacement_field = filter.Execute(out)

    return displacement_field


def register_to_atlas(tissue, section, label, class_map_path):
    """Uses demons registration to register a tissue section to the atlas"""
    with open(class_map_path, "rb") as f:
        classMap = pickle.load(f)
        classMap[997] = {"index": 1326, "name": "undefined", "color": [0, 0, 0]}
        classMap[0] = {"index": 1327, "name": "Lost in Warp", "color": [0, 0, 0]}

    scaled_atlas = Image.fromarray(section)
    scaled_atlas = np.array(scaled_atlas)
    scaled_atlas = (scaled_atlas / 256).astype(np.uint8)
    scaled_label = Image.fromarray(label)
    scaled_label = np.array(scaled_label)

    fixed = sitk.GetImageFromArray(tissue, isVector=False)
    moving = sitk.GetImageFromArray(scaled_atlas, isVector=False)
    label = sitk.GetImageFromArray(scaled_label, isVector=False)

    matcher = sitk.HistogramMatchingImageFilter()
    if fixed.GetPixelID() in (sitk.sitkUInt8, sitk.sitkInt8):
        matcher.SetNumberOfHistogramLevels(128)
    else:
        matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    moving = matcher.Execute(moving, fixed)

    demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()
    demons.SetNumberOfIterations(2000)
    demons.SetSmoothDisplacementField(True)
    demons.SetStandardDeviations(1.5)
    displacement_field = demons.Execute(fixed, moving)

    transformation = sitk.DisplacementFieldTransform(displacement_field)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(transformation)
    resampler.SetOutputPixelType(sitk.sitkUInt32)
    resampler.SetDefaultPixelValue(0)

    resampled_label = resampler.Execute(label)
    resampled_atlas = resampler.Execute(moving)

    color_label = np.zeros(
        (resampled_label.GetSize()[1], resampled_label.GetSize()[0], 3)
    )
    for i in range(resampled_label.GetSize()[1]):
        for j in range(resampled_label.GetSize()[0]):
            try:
                color_label[i, j, :] = classMap[resampled_label.GetPixel(j, i)]["color"]
            except:
                pass

    resampled_label = sitk.GetArrayFromImage(resampled_label).astype(np.uint32)
    resampled_atlas = sitk.GetArrayFromImage(resampled_atlas).astype(np.uint8)
    return resampled_label, resampled_atlas, color_label
