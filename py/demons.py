import SimpleITK as sitk
import numpy as np
from PIL import Image
import pickle
# Check number of cores available
import multiprocessing
# Set sitk to use cores - 2
sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(multiprocessing.cpu_count() - 2)

def log_progress(registration_method):
    print(f'Level: {registration_method.GetCurrentLevel()}', flush=True)
    print(f'Metric value: {registration_method.GetMetricValue()}', flush=True)
    print(f'Learning rate: {registration_method.GetOptimizerLearningRate()}', flush=True)


def mean_squares_registration(fixed, moving, current_tx=None):
    '''Performs mean squares registration between two images, captures size dynamics well for initial registration'''
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)

    if current_tx is None:
        initial_tx = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.AffineTransform(fixed.GetDimension())
        )
    else:
        initial_tx = current_tx

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMeanSquares()
    registration_method.SetInitialTransform(initial_tx, inPlace=True)
    registration_method.SetShrinkFactorsPerLevel([5, 3, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel([3, 2, 1, 1])
    
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=500,
        estimateLearningRate=registration_method.EachIteration,
    )

    registration_method.SetOptimizerScalesFromPhysicalShift()
    #  registration_method.AddCommand(sitk.sitkIterationEvent, lambda: log_progress(registration_method))

    final_tx = registration_method.Execute(fixed, moving)

    return final_tx

def mutual_information_registration(fixed, moving, current_tx=None):
    '''Sequentially performs JMHI registration, captures shape dynamics well for final registration'''
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)
    if current_tx is None:
        initial_tx = sitk.CenteredTransformInitializer(
            fixed, moving, sitk.AffineTransform(fixed.GetDimension())
        )
    else:
        initial_tx = current_tx

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsJointHistogramMutualInformation()
    registration_method.SetInitialTransform(initial_tx, inPlace=True)
    registration_method.SetShrinkFactorsPerLevel([5, 3, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel([3, 2, 1, 1])

    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=500,
        estimateLearningRate=registration_method.EachIteration,
    )

    registration_method.SetOptimizerScalesFromPhysicalShift()
    # registration_method.AddCommand(sitk.sitkIterationEvent, lambda: log_progress(registration_method))

    final_tx = registration_method.Execute(fixed, moving)

    return final_tx

def ants_registration(fixed, moving, current_tx=None):
    ''' 
    Performs ANTS registration between two images, captures size dynamics well for initial registration

    Intialzed as a displacement field transform, uses the following parameters
    '''

    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)

    if current_tx is None:
        displacement_field = sitk.Image(fixed.GetSize(), sitk.sitkVectorFloat64)
        displacement_field.CopyInformation(fixed)
        initial_tx = sitk.DisplacementFieldTransform(displacement_field)
        del displacement_field
        initial_tx.SetSmoothingGaussianOnUpdate(
            varianceForUpdateField=0.0, varianceForTotalField=1.5
        )
    else:
        # convert current transform to displacement field
        field_filter = sitk.TransformToDisplacementFieldFilter()
        field_filter.SetReferenceImage(fixed)
        curr_field = field_filter.Execute(current_tx)
        initial_tx = sitk.DisplacementFieldTransform(curr_field)

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetInitialTransform(initial_tx, inPlace=True)
    registration_method.SetMetricAsANTSNeighborhoodCorrelation(5)
    
    registration_method.SetShrinkFactorsPerLevel([5, 3, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel([3, 2, 1, 1])

    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=1500,
        estimateLearningRate=registration_method.EachIteration,
    )

    registration_method.SetOptimizerScalesFromPhysicalShift()

    registration_method.Execute(fixed, moving)

    return initial_tx

def demons_registration(fixed, moving, current_tx=None):
    '''Performs demons registration between two images, captures shape dynamics well for final registration'''

    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)

    demons = sitk.FastSymmetricForcesDemonsRegistrationFilter()

    demons.SetNumberOfIterations(250)
    demons.SetSmoothDisplacementField(True)
    demons.SetStandardDeviations(1.5)

    if current_tx is None:
        final_field = demons.Execute(fixed, moving)
    else:
        # convert current transform to displacement field
        field_filter = sitk.TransformToDisplacementFieldFilter()
        field_filter.SetReferenceImage(fixed)
        curr_field = field_filter.Execute(current_tx)
        final_field = demons.Execute(fixed, moving, curr_field)
    # convert deformation field to displacement field
    tx = sitk.DisplacementFieldTransform(final_field)

    return tx

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

    tx_initial = mean_squares_registration(fixed, moving)
    tx_middle = mutual_information_registration(fixed, moving, tx_initial)
    tx_final = ants_registration(fixed, moving, tx_middle)
    tx_demons = demons_registration(fixed, moving, tx_final)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(tx_demons)
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
