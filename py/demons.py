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
    # Pad
    padding_size = 64
    fixed = sitk.ConstantPad(fixed, (padding_size, padding_size))
    moving = sitk.ConstantPad(moving, (padding_size, padding_size))
    # Cast
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)

    # Affine
    initialTx = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(fixed.GetDimension())
    )

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(32)
    R.SetOptimizerAsGradientDescent(
        learningRate=0.001,
        numberOfIterations=500,
        convergenceMinimumValue=1e-8,
        convergenceWindowSize=20,
    )
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    R.SetInitialTransform(initialTx)
    R.SetInterpolator(sitk.sitkLinear)

    outTx1 = R.Execute(fixed, moving)

    # Resample the moving image using the initial transformation
    resampled_moving = sitk.Resample(
        moving, fixed, outTx1, sitk.sitkLinear, 0.0, sitk.sitkFloat32
    )
    # B-spline
    transformDomainMeshSize = [4] * fixed.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed, transformDomainMeshSize)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(tx, inPlace=False)

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


def resize_image_nearest_neighbor(input_image, new_size):
    """
    Resize an image using nearest-neighbor interpolation, maintaining the original data type.

    Parameters:
        input_image (SimpleITK.Image): The input image to be resized.
        new_size (tuple or list): The desired size (in pixels) as (height, width, [depth]).

    Returns:
        SimpleITK.Image: The resized image, maintaining the original data type.
    """

    # Calculate the new spacing based on old spacing and old and new sizes
    input_image = sitk.GetImageFromArray(input_image)
    original_size = input_image.GetSize()
    original_spacing = input_image.GetSpacing()
    new_spacing = [
        float(orig_space) * float(orig_size) / float(new_dim)
        for orig_space, orig_size, new_dim in zip(
            original_spacing, original_size, new_size
        )
    ]

    # Set up the resampler with nearest neighbor interpolation, original data type is maintained by default
    resampler = sitk.ResampleImageFilter()
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputPixelType(input_image.GetPixelIDValue())
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(input_image.GetOrigin())
    resampler.SetOutputDirection(input_image.GetDirection())

    # Apply the resampling operation
    resized_image = resampler.Execute(input_image)
    resized_image = sitk.GetArrayFromImage(resized_image)
    return resized_image


def register_to_atlas(tissue, section, label, structure_map_path):
    """
    Register a section to the atlas using sitk.

    Args:
        tissue (numpy.ndarray): The tissue image.
        section (numpy.ndarray): The section image.
        label (numpy.ndarray): The label image.
        class_map_path (str): The path to the class map pickle file.

    Returns:
        numpy.ndarray: The registered label image.
        numpy.ndarray: The registered atlas image.
        numpy.ndarray: The color label image.
    """

    with open(structure_map_path, "rb") as f:
        structure_map = pickle.load(f)

    fixed = sitk.GetImageFromArray(tissue, isVector=False)
    moving = sitk.GetImageFromArray(section, isVector=False)
    label = sitk.GetImageFromArray(label, isVector=False)
    # resize atlas to match section
    moving = match_histograms(fixed, moving)
    tx = multimodal_registration(fixed, moving)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(tx)
    resampler.SetOutputPixelType(sitk.sitkUInt32)
    resampler.SetDefaultPixelValue(0)
    resampled_label = resampler.Execute(label)
    resampler.SetOutputPixelType(sitk.sitkUInt8)
    resampled_atlas = resampler.Execute(moving)

    color_label = np.zeros(
        (resampled_label.GetSize()[1], resampled_label.GetSize()[0], 3), dtype=np.uint8
    )

    for i in range(resampled_label.GetSize()[1]):
        for j in range(resampled_label.GetSize()[0]):
            try:
                color_label[i, j, :] = structure_map[resampled_label.GetPixel(j, i)][
                    "color"
                ]
            except:
                pass

    resampled_label = sitk.GetArrayFromImage(resampled_label)
    resampled_atlas = sitk.GetArrayFromImage(resampled_atlas)

    return resampled_label, resampled_atlas, color_label
