import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from math import pi
import pickle


def multi_stage_registration(fixed, moving):
    """Uses exhaustive search to find the best transformation"""

    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)

    initialTx = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(fixed.GetDimension())
    )

    R = sitk.ImageRegistrationMethod()

    R.SetShrinkFactorsPerLevel([3, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 1])

    R.SetMetricAsJointHistogramMutualInformation(5)
    R.MetricUseFixedImageGradientFilterOff()

    R.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=250,
        estimateLearningRate=R.EachIteration,
    )
    R.SetOptimizerScalesFromPhysicalShift()

    R.SetInitialTransform(initialTx)

    R.SetInterpolator(sitk.sitkLinear)

    outTx1 = R.Execute(fixed, moving)

    # Mean Squares
    R.SetMetricAsMeanSquares()

    R.SetInitialTransform(outTx1)

    R.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        estimateLearningRate=R.EachIteration,
    )
    outTx2 = R.Execute(fixed, moving)

    displacementField = sitk.Image(fixed.GetSize(), sitk.sitkVectorFloat64)
    displacementField.CopyInformation(fixed)
    displacementTx = sitk.DisplacementFieldTransform(displacementField)
    del displacementField
    displacementTx.SetSmoothingGaussianOnUpdate(
        varianceForUpdateField=0.0, varianceForTotalField=1.5
    )

    R.SetMovingInitialTransform(outTx2)
    R.SetInitialTransform(displacementTx, inPlace=True)

    R.SetMetricAsANTSNeighborhoodCorrelation(4)
    R.MetricUseFixedImageGradientFilterOff()

    R.SetShrinkFactorsPerLevel([3, 2, 1])
    R.SetSmoothingSigmasPerLevel([2, 1, 1])

    R.SetOptimizerScalesFromPhysicalShift()

    R.Execute(fixed, moving)

    # Demons

    R.SetMetricAsDemons()
    R.SetMetricSamplingPercentage(0.50)

    R.Execute(fixed, moving)

    compositeTx = sitk.CompositeTransform([outTx2, displacementTx])

    return compositeTx


def match_histograms(src, target):
    """Match the src histogram to the target using sitk"""

    matcher = sitk.HistogramMatchingImageFilter()

    if src.GetPixelID() in (sitk.sitkUInt8, sitk.sitkInt8):
        matcher.SetNumberOfHistogramLevels(128)
    else:
        matcher.SetNumberOfHistogramLevels(1024)

    matcher.SetNumberOfMatchPoints(10)
    matcher.ThresholdAtMeanIntensityOn()
    return matcher.Execute(src, target)


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

    moving = match_histograms(moving, fixed)

    transformation = multi_stage_registration(fixed, moving)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(transformation)
    resampler.SetOutputPixelType(sitk.sitkUInt32)
    resampler.SetDefaultPixelValue(0)

    resampled_atlas = resampler.Execute(moving)
    resampled_label = resampler.Execute(label)

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
