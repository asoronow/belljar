import SimpleITK as sitk
import numpy as np
from PIL import Image
import pickle

# Check number of cores available
import multiprocessing

# Set sitk to use cores - 2
sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(multiprocessing.cpu_count() - 2)


def match_histograms(fixed, moving):
    """Match the moving histogram to the fixed using sitk"""
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(10)
    matcher.ThresholdAtMeanIntensityOn()
    return matcher.Execute(moving, fixed)


def multimodal_registration(fixed, moving):
    # Initial alignment using an affine transformation
    initialTx = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(fixed.GetDimension())
    )

    # Pad
    padding_size = [32] * fixed.GetDimension()
    fixed = sitk.ConstantPad(fixed, padding_size)
    moving = sitk.ConstantPad(moving, padding_size)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(25)
    R.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-10,
        convergenceWindowSize=5,
        estimateLearningRate=R.EachIteration
    )
    R.SetOptimizerScalesFromPhysicalShift()

    R.SetShrinkFactorsPerLevel([8, 4, 2, 1])
    R.SetSmoothingSigmasPerLevel([3, 2, 1, 0])

    R.SetInitialTransform(initialTx)
    R.SetInterpolator(sitk.sitkLinear)

    outTx1 = R.Execute(fixed, moving)

    # Resample the moving image using the initial transformation
    resampled_moving = sitk.Resample(
        moving, fixed, outTx1, sitk.sitkLinear, 0.0, moving.GetPixelID()
    )

    # B-spline registration
    transformDomainMeshSize = [6] * fixed.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed, transformDomainMeshSize)

    R.SetMetricAsMattesMutualInformation(25)
    R.SetShrinkFactorsPerLevel([8, 4, 2, 1])
    R.SetSmoothingSigmasPerLevel([3, 2, 1, 0])
    R.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=1000,
        convergenceMinimumValue=1e-10,
        convergenceWindowSize=10,
        estimateLearningRate=R.EachIteration
    )
    R.SetOptimizerScalesFromPhysicalShift()

    R.SetInitialTransform(tx, inPlace=False)
    R.SetInterpolator(sitk.sitkNearestNeighbor)

    outTx2 = R.Execute(fixed, resampled_moving)

    # Combine the transformations: Affine followed by B-spline.
    composite_transform = sitk.CompositeTransform(outTx1)
    composite_transform.AddTransform(outTx2)

    return composite_transform


def register_to_atlas(tissue, section, label, class_map_path):
    """Uses deformable registration to register a tissue section to the atlas"""
    with open(class_map_path, "rb") as f:
        classMap = pickle.load(f)
        classMap[997] = {"index": 1326, "name": "undefined", "color": [0, 0, 0]}
        classMap[0] = {"index": 1327, "name": "Lost in Warp", "color": [0, 0, 0]}

    fixed = sitk.GetImageFromArray(tissue, isVector=False)
    moving = sitk.GetImageFromArray(section, isVector=False)
    label = sitk.GetImageFromArray(label, isVector=False)

    moving = sitk.Cast(moving, sitk.sitkFloat32)
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)

    moving = match_histograms(fixed, moving)

    tx = multimodal_registration(fixed, moving)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(tx)
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
