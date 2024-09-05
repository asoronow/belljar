import cv2
import SimpleITK as sitk
import numpy as np
import nrrd
from pathlib import Path
import argparse
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog
from slice_atlas import slice_3d_volume
from skimage.filters import difference_of_gaussians, sobel, unsharp_mask, threshold_otsu
from skimage.morphology import binary_closing, remove_small_objects

ACTIVE_SLIDER = 0
X_ANGLE = 0
Y_ANGLE = 0



def match_histograms(fixed, moving):
    """
    Match the moving histogram to the fixed using sitk
    Args:
        fixed (sitk.Image): The fixed image.
        moving (sitk.Image): The moving image.
    Returns:
        sitk.Image: The matched moving image.
    """
    matcher = sitk.HistogramMatchingImageFilter()
    matcher.SetNumberOfHistogramLevels(1024)
    matcher.SetNumberOfMatchPoints(10)
    matcher.ThresholdAtMeanIntensityOn()
    return matcher.Execute(moving, fixed)

def visualize_registration(fixed, moving, transform, title=""):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(transform)

    out = resampler.Execute(moving)
    # Convert to numpy arrays for visualization
    fixed_np = sitk.GetArrayViewFromImage(fixed)
    moving_np = sitk.GetArrayViewFromImage(moving)
    out_np = sitk.GetArrayViewFromImage(out)
    
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 3, 1)
    plt.title("Fixed")
    plt.imshow(fixed_np, cmap="viridis")
    plt.subplot(1, 3, 2)
    plt.title("Moving")
    plt.imshow(moving_np, cmap="viridis")
    plt.subplot(1, 3, 3)
    plt.title("Transformed Moving")
    plt.imshow(out_np, cmap="viridis")
    plt.suptitle(title)
    plt.show()


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
    # Normalize the image
def multimodal_registration(fixed, moving):
    fixed = preprocess_image(fixed)
    moving = preprocess_image(moving)
    # Affine
    initialTx = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(fixed.GetDimension())
    )

    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation()
    R.SetOptimizerAsGradientDescent(
        learningRate=0.01,
        numberOfIterations=300,
        convergenceMinimumValue=1e-8,
        convergenceWindowSize=20,
    )
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    R.SetSmoothingSigmasPerLevel(smoothingSigmas=[3, 2, 0])
    R.SetInitialTransform(initialTx)
    R.SetInterpolator(sitk.sitkLinear)

    outTx1 = R.Execute(fixed, moving)

    # Resample the moving image using the initial transformation
    resampled_moving = sitk.Resample(
        moving, fixed, outTx1, sitk.sitkLinear, 0.0, sitk.sitkFloat32
    )
    # B-spline
    transformDomainMeshSize = [6] * fixed.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed, transformDomainMeshSize)
    R.SetMetricAsANTSNeighborhoodCorrelation(11)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(tx, inPlace=False)
    R.SetOptimizerAsGradientDescent(
        learningRate=0.001,
        numberOfIterations=300,
        convergenceMinimumValue=1e-10,
        convergenceWindowSize=20,
    )
    outTx2 = R.Execute(fixed, resampled_moving)

    # Combine the transformations: Affine followed by B-spline.
    composite_transform = sitk.CompositeTransform(outTx1)
    composite_transform.AddTransform(outTx2)

    return composite_transform
class AtlasSliceViewer:
    def __init__(self, atlas_path):
        self.atlas_path = Path(atlas_path).expanduser()
        self.atlas, _ = nrrd.read(self.atlas_path)
        self.active_slider = 0
        self.x_angle = 0
        self.y_angle = 0
        self.init_ui()

    def init_ui(self):
        self.root = tk.Tk()
        self.root.title("Atlas Viewer")
        self.sliders = self.make_sliders(self.max_min_atlas(self.atlas))
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(self.slice_3d_volume(0, 0, 0, 0), cmap='gray')
        self.btn_save = tk.Button(self.root, text="Save as PNG", command=self.save_image)
        self.btn_save.pack()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=1)
        self.root.mainloop()

    def max_min_atlas(self, atlas):
        """
        Create a function that takes an atlas and returns the max and min of each dimension
        """
        return [atlas.shape[i] - 1 for i in range(3)]

    def make_sliders(self, max_min_atlas):
        """
        Create a function that takes max and min of atlas and returns three sliders
        """
        sliders = []
        for i in range(3):
            slider = tk.Scale(
                self.root,
                from_=0,
                to=max_min_atlas[i],
                name=str(i),
                orient=tk.HORIZONTAL,
                command=lambda value, i=i: self.update_slice(i)
            )
            slider.pack()
            sliders.append(slider)
        
        x_slider = tk.Scale(
            self.root,
            from_=-5.0,
            to=5.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            command=lambda value: self.update_slice(3)
        )
        y_slider = tk.Scale(
            self.root,
            from_=-5.0,
            to=5.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            command=lambda value: self.update_slice(4)
        )
        x_slider.pack()
        y_slider.pack()
        sliders.append(x_slider)
        sliders.append(y_slider)

        return sliders

    def update_slice(self, slider_num):
        """
        Create a function that will update the displayed slice when the sliders are moved
        """
        if slider_num not in [3, 4]:
            self.active_slider = slider_num

            for i, slider in enumerate(self.sliders):
                if i != self.active_slider:
                    slider.set(0)
        
        self.x_angle = self.sliders[3].get()
        self.y_angle = self.sliders[4].get()
        current = self.sliders[self.active_slider].get()
        self.ax.clear()
        self.ax.imshow(cv2.rotate(self.slice_3d_volume(current, self.x_angle, self.y_angle, self.active_slider), cv2.ROTATE_90_CLOCKWISE), cmap='gray')
        self.fig.canvas.draw()

    def save_image(self):
        """
        Save the current slice as a PNG file
        """
        file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if file_path:
            cv2.imwrite(file_path, cv2.rotate(self.slice_3d_volume(self.sliders[self.active_slider].get(), self.x_angle, self.y_angle, self.active_slider), cv2.ROTATE_90_CLOCKWISE))

if __name__ == "__main__":
    """
    A Generic script for testing image alignment and registration on a single sample image.
    Useful for tuning the alignment and registration parameters.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("fixed", help="path to fixed image")
    parser.add_argument("moving", help="path to moving image")
    args = parser.parse_args()

    fixed = cv2.imread(args.fixed, cv2.IMREAD_GRAYSCALE)
    moving = cv2.imread(args.moving, cv2.IMREAD_GRAYSCALE)

    # Padding
    fixed = cv2.copyMakeBorder(fixed, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=0)
    moving = cv2.copyMakeBorder(moving, 25, 25, 25, 25, cv2.BORDER_CONSTANT, value=0)
    # resize
    fixed = cv2.resize(fixed, (256, 256))
    moving = cv2.resize(moving, (256, 256))
    
    # Cast to sitk image
    fixed = sitk.Cast(sitk.GetImageFromArray(fixed), sitk.sitkFloat32)
    moving = sitk.Cast(sitk.GetImageFromArray(moving), sitk.sitkFloat32)
    
    # Histogram matching
    moving = match_histograms(fixed, moving)

    # Transform 
    tx = multimodal_registration(fixed, moving)

    # Resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(sitk.sitkUInt8)
    resampler.SetTransform(tx)
    aligned = resampler.Execute(moving)
    aligned = sitk.GetArrayFromImage(aligned)
    fixed = sitk.GetArrayFromImage(fixed).astype(np.uint8)
    moving = sitk.GetArrayFromImage(moving).astype(np.uint8)
    # Visualize
    fig, axes = plt.subplots(1, 4, figsize=(12, 6))
    axes[0].set_title("Original Image")
    axes[0].imshow(fixed, cmap="gray")
    axes[1].set_title("Atlas Match Image")
    axes[1].imshow(moving, cmap="gray")
    axes[2].set_title("Aligned Image")
    axes[2].imshow(aligned, cmap="gray")
    axes[3].set_title("Overlaid (red = Original, blue = Aligned)")
    fixed_color = cv2.cvtColor(fixed, cv2.COLOR_GRAY2BGR)
    aligned_color = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
    red_fixed = cv2.cvtColor(fixed_color, cv2.COLOR_BGR2RGB) * (1, 0, 0)
    blue_aligned = cv2.cvtColor(aligned_color, cv2.COLOR_BGR2RGB) * (0, 0, 1)
    overlaid = cv2.addWeighted(red_fixed, 0.5, blue_aligned, 0.5, 0)
    axes[3].imshow(overlaid)
    plt.show()