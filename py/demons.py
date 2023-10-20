import SimpleITK as sitk
import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2

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
    padding_size = [64] * fixed.GetDimension()
    fixed = sitk.ConstantPad(fixed, padding_size)
    moving = sitk.ConstantPad(moving, padding_size)

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(32)
    R.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=5,
        estimateLearningRate=R.EachIteration,
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

    R.SetMetricAsMattesMutualInformation(32)
    R.SetShrinkFactorsPerLevel([8, 4, 2, 1])
    R.SetSmoothingSigmasPerLevel([3, 2, 1, 0])
    R.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=5,
        estimateLearningRate=R.EachIteration,
    )
    R.SetOptimizerScalesFromPhysicalShift()

    R.SetInitialTransform(tx, inPlace=False)
    R.SetInterpolator(sitk.sitkNearestNeighbor)

    outTx2 = R.Execute(fixed, resampled_moving)

    # DEBUG: Quiver plot, uncomment to see plots of each transformation
    # Combine the transformations: Affine followed by B-spline.
    composite_transform = sitk.CompositeTransform(outTx1)
    composite_transform.AddTransform(outTx2)

    # Calculate displacement field from the composite transform
    # displacement_field = sitk.TransformToDisplacementField(
    #     composite_transform,
    #     sitk.sitkVectorFloat64,
    #     fixed.GetSize(),
    #     fixed.GetOrigin(),
    #     fixed.GetSpacing(),
    #     fixed.GetDirection(),
    # )

    # # Convert the displacement field to numpy arrays
    # displacements = sitk.GetArrayFromImage(displacement_field)

    # # Get the x and y components of the displacements
    # dy, dx = displacements[..., 0], displacements[..., 1]

    # # For visualization purposes, it may be helpful to sample every k'th point to avoid overcrowding in the quiver plot
    # k = 10
    # grid_y, grid_x = np.mgrid[
    #     0 : displacements.shape[0] : k, 0 : displacements.shape[1] : k
    # ]
    # disp_y, disp_x = dy[::k, ::k], dx[::k, ::k]

    # # Use quiver plot to visualize the displacements
    # plt.figure(figsize=(10, 10))
    # plt.imshow(sitk.GetArrayFromImage(fixed), cmap="gray")
    # plt.quiver(grid_x, grid_y, disp_x, disp_y, angles="xy", scale_units="xy", color="r")
    # # turn off axis to remove clutter
    # plt.axis("off")
    # plt.show()

    return composite_transform


def resize_image_to_original(image, original_size):
    """
    Resize the image back to its original size.

    Parameters:
        image: Resized image.
        original_size: Tuple of the original height and width.

    Returns:
        Image resized back to its original dimensions.
    """
    original_height, original_width = original_size

    # Resize the image using nearest neighbor interpolation
    resized_image = cv2.resize(
        image, (original_width, original_height), interpolation=cv2.INTER_NEAREST
    )

    return resized_image


def resize_image_with_aspect_ratio(image, target_width):
    """
    Resize the image to the specified width while maintaining its aspect ratio.

    Parameters:
        image: Original image to be resized.
        target_width: Desired width of the resized image.

    Returns:
        Resized image.
    """
    original_height, original_width = image.shape[:2]

    # Calculate the aspect ratio
    aspect_ratio = float(original_height) / original_width

    # Calculate the new height using the aspect ratio
    new_height = int(target_width * aspect_ratio)

    # Resize the image using nearest neighbor interpolation
    resized_image = cv2.resize(
        image, (target_width, new_height), interpolation=cv2.INTER_NEAREST
    )

    return resized_image


def register_to_atlas(tissue, section, label, class_map_path):
    """Uses deformable registration to register a tissue section to the atlas"""
    with open(class_map_path, "rb") as f:
        classMap = pickle.load(f)
        classMap[997] = {"index": 1326, "name": "undefined", "color": [0, 0, 0]}
        classMap[0] = {"index": 1327, "name": "Lost in Warp", "color": [0, 0, 0]}

    tissue_original_size = tissue.shape[:2]
    tissue = resize_image_with_aspect_ratio(tissue, 1024)
    section = resize_image_with_aspect_ratio(section, 1024)
    label = resize_image_with_aspect_ratio(label.astype(np.int32), 1024)

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

    resampled_label = sitk.GetArrayFromImage(resampled_label).astype(np.int32)
    resampled_atlas = sitk.GetArrayFromImage(resampled_atlas).astype(np.uint8)

    resampled_label = resize_image_to_original(resampled_label, tissue_original_size)
    resampled_atlas = resize_image_to_original(resampled_atlas, tissue_original_size)
    color_label = resize_image_to_original(color_label, tissue_original_size)

    return resampled_label, resampled_atlas, color_label
