import SimpleITK as sitk
import numpy as np
import pickle
import matplotlib.pyplot as plt
import cv2
from skimage.filters import sobel, gaussian, difference_of_gaussians

# Check number of cores available
import multiprocessing

# Set sitk to use cores - 2
sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(multiprocessing.cpu_count() - 2)


def match_histograms(to_match, match_to):
    """
    Match the to_match histogram to the match_to using sitk
    Args:
        to_match (sitk.Image): The image to be matched.
        match_to (sitk.Image): The image to be matched to.
    Returns:
        sitk.Image: The matched image.
    """
    # make sure fixed and moving are sitk images
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(10)
    matcher.ThresholdAtMeanIntensityOn()
    return matcher.Execute(to_match, match_to)


# DEBUG: Quiver plot, uncomment to see plots of each transformation
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

def preprocess_image(image):
    """
    Preprocess the image to enhance features.
    """
     # Convert SimpleITK image to numpy array
    image_array = sitk.GetArrayFromImage(sitk.Cast(image, sitk.sitkUInt8))
    blurred = cv2.GaussianBlur(image_array, (5, 5), 0)
    edges = sobel(blurred)
    # normalize
    edges = (edges - np.min(edges)) / (np.max(edges) - np.min(edges))
    edges = edges.astype(np.float32)
    edges = sitk.GetImageFromArray(edges)

    return edges
    

def multimodal_registration(fixed, moving):
    fixed = preprocess_image(fixed)
    moving = preprocess_image(moving)
    # Affine transformation
    initialTx = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(fixed.GetDimension())
    )

    # Set up the image registration method for the affine transformation
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsCorrelation()
    R.SetOptimizerAsGradientDescent(
        learningRate=0.01,
        numberOfIterations=100,
        convergenceMinimumValue=1e-12,
        convergenceWindowSize=10,
    )
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    R.SetInitialTransform(initialTx)
    R.SetInterpolator(sitk.sitkLinear)

    outTx1 = R.Execute(fixed, moving)

    # Resample the moving image using the affine transformation
    resampled_moving = sitk.Resample(
        moving, fixed, outTx1, sitk.sitkLinear, 0.0, moving.GetPixelID()
    )

    # B-spline transformation
    transformDomainMeshSize = [4] * fixed.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed, transformDomainMeshSize)
    R.SetInitialTransform(tx, inPlace=False)
    R.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)  # Metric reset for B-spline
    R.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    R.SetOptimizerAsGradientDescent(
        learningRate=0.0001,
        numberOfIterations=100,
        convergenceMinimumValue=1e-14,
        convergenceWindowSize=20,
    )
    R.SetOptimizerScalesFromPhysicalShift()

    outTx2 = R.Execute(fixed, resampled_moving)

    # Combine the transformations: Affine followed by B-spline.
    composite_transform = sitk.CompositeTransform(fixed.GetDimension())
    composite_transform.AddTransform(outTx1)
    composite_transform.AddTransform(outTx2)

    return composite_transform

def resize_image_to_width(image, target_width):
    """
    Resize an image to a target width while maintaining the aspect ratio.

    Parameters:
    - image: The input SimpleITK image.
    - target_width: The desired width of the image after resizing.

    Returns:
    - The resized SimpleITK image.
    """
    # Get the original size and spacing of the image
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    # Calculate the new height maintaining the aspect ratio
    aspect_ratio = original_size[1] / original_size[0]
    new_height = int(target_width * aspect_ratio)

    # Calculate the new spacing to maintain the aspect ratio
    new_spacing = [
        original_size[0] / target_width * original_spacing[0],
        original_size[1] / new_height * original_spacing[1],
    ]

    # Set up the resampling filter
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize((target_width, new_height))
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)

    # Perform the resampling
    resized_image = resampler.Execute(image)

    return resized_image


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

    tissue_resized = cv2.resize(tissue, (360, 360))
    section_resized = cv2.resize(section, (360, 360))
    label = resize_image_nearest_neighbor(label, (360, 360))
    fixed = sitk.GetImageFromArray(tissue_resized, isVector=False)
    
    for i in range(section_resized.shape[0]):
        for j in range(section_resized.shape[1]):
            if "layer 4" in structure_map[label[i, j]]["name"].lower():
                section_resized[i, j] = min(section_resized[i, j] + 15, 255)
            elif "layer 5" in structure_map[label[i, j]]["name"].lower():
                section_resized[i, j] = max(section_resized[i, j] - 7, 0)


    moving = sitk.GetImageFromArray(section_resized, isVector=False)
    label = sitk.GetImageFromArray(label, isVector=False)
    fixed = match_histograms(fixed, moving)    
    # cast to float 32
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)
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
   
   # conver color label to cv2
    color_label = cv2.cvtColor(color_label, cv2.COLOR_RGB2BGR)
    resampled_label = sitk.GetArrayFromImage(resampled_label)
    resampled_atlas = sitk.GetArrayFromImage(resampled_atlas)
    # resize atlas back to original size
    resampled_atlas = cv2.resize(resampled_atlas, tissue.shape[:2][::-1])
    color_label = cv2.resize(color_label, tissue.shape[:2][::-1])
    # convert color label back to rgb
    color_label = cv2.cvtColor(color_label, cv2.COLOR_BGR2RGB)
    resampled_label = resize_image_nearest_neighbor(resampled_label, tissue.shape[:2][::-1])

    return resampled_label, resampled_atlas, color_label
