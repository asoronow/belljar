import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import nrrd
from PIL import Image
import pickle

if __name__ == "__main__":
    nrrd_path = "C:\\Users\\Alec\\Projects\\aba-nrrd\\raw"
    classMap_path = "C:\\Users\\Alec\\Projects\\aba-nrrd\\raw\\classMap.pkl"
    with open(classMap_path, "rb") as f:
        classMap = pickle.load(f)
        classMap[997] = {"index": 1326, "name": "undefined", "color": [0, 0, 0]}
        classMap[0] = {"index": 1327, "name": "Lost in Warp", "color": [0, 0, 0]}
    zero_angle_atlas = nrrd.read(nrrd_path + "\\r_nissl_0.nrrd")[0]
    zero_angle_labels = nrrd.read(nrrd_path + "\\r_annotation_0.nrrd")[0]
    dapi_image_path = "C:\\Users\\Alec\\Downloads\\DAPI\\M496_s013.png"
    dapi_image = Image.open(dapi_image_path)
    dapi_image = np.array(dapi_image)
    # get the middle section
    zero_angle_atlas_image = zero_angle_atlas[758, :, :]
    zero_angle_atlas_label = zero_angle_labels[758, :, :]
    zero_angle_atlas_image = Image.fromarray(zero_angle_atlas_image)
    zero_angle_atlas_image = zero_angle_atlas_image.resize(
        (dapi_image.shape[1], dapi_image.shape[0])
    )
    zero_angle_atlas_image = np.array(zero_angle_atlas_image)
    zero_angle_atlas_label = Image.fromarray(zero_angle_atlas_label)
    zero_angle_atlas_label = zero_angle_atlas_label.resize(
        (dapi_image.shape[1], dapi_image.shape[0]), resample=Image.Resampling.NEAREST
    )
    zero_angle_atlas_label = np.array(zero_angle_atlas_label)

    def command_iteration(filter):
        print(f"{filter.GetElapsedIterations():3} = {filter.GetMetric():10.5f}")

    fixed = sitk.ReadImage(dapi_image_path, sitk.sitkFloat32)

    moving = sitk.GetImageFromArray(zero_angle_atlas_image, isVector=False)
    label = sitk.GetImageFromArray(zero_angle_atlas_label, isVector=False)

    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(7)
    matcher.ThresholdAtMeanIntensityOn()
    moving = matcher.Execute(moving, fixed)

    # The basic Demons Registration Filter
    # Note there is a whole family of Demons Registration algorithms included in
    # SimpleITK
    demons = sitk.DemonsRegistrationFilter()
    demons.SetNumberOfIterations(2000)
    # Standard deviation for Gaussian smoothing of displacement field
    demons.SetStandardDeviations(4.0)

    demons.AddCommand(sitk.sitkIterationEvent, lambda: command_iteration(demons))

    displacementField = demons.Execute(fixed, moving)

    print("-------")
    print(f"Number Of Iterations: {demons.GetElapsedIterations()}")
    print(f" RMS: {demons.GetRMSChange()}")

    outTx = sitk.DisplacementFieldTransform(displacementField)

    # sitk.WriteTransform(outTx, sys.argv[3])

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(outTx)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(sitk.sitkUInt32)
    out = resampler.Execute(moving)
    out_label = resampler.Execute(label)

    color_label = np.zeros((out_label.GetSize()[1], out_label.GetSize()[0], 3))
    for i in range(out_label.GetSize()[1]):
        for j in range(out_label.GetSize()[0]):
            try:
                color_label[i, j, :] = classMap[out_label.GetPixel(j, i)]["color"]
            except:
                pass

    color_label = Image.fromarray(color_label.astype(np.uint8))

    simg1 = sitk.Cast(sitk.RescaleIntensity(fixed), sitk.sitkUInt8)
    simg2 = sitk.Cast(sitk.RescaleIntensity(out), sitk.sitkUInt8)
    # Use the // floor division operator so that the pixel type is
    # the same for all three images which is the expectation for
    # the compose filter.
    color_label.save("C:\\Users\\Alec\\Downloads\\DAPI\\color_label_p.png")
    out = Image.fromarray(sitk.GetArrayFromImage(out).astype(np.uint8))
    out.save("C:\\Users\\Alec\\Downloads\\DAPI\\out_p.png")
    sitk.Show(out_label, "Before Registration")
