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

ACTIVE_SLIDER = 0
X_ANGLE = 0
Y_ANGLE = 0

def multimodal_registration(fixed, moving):
    # Cast
    fixed = sitk.Cast(sitk.GetImageFromArray(fixed), sitk.sitkFloat32)
    moving = sitk.Cast(sitk.GetImageFromArray(moving), sitk.sitkFloat32)

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
    transformDomainMeshSize = [4] * fixed.GetDimension()
    tx = sitk.BSplineTransformInitializer(fixed, transformDomainMeshSize)
    R.SetMetricAsANTSNeighborhoodCorrelation(16)
    R.SetOptimizerScalesFromPhysicalShift()
    R.SetInitialTransform(tx, inPlace=False)
    R.SetOptimizerAsGradientDescent(
        learningRate=0.01,
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
    blurred_moving = cv2.GaussianBlur(moving, (5, 5), 0)
    blurred_fixed = cv2.GaussianBlur(fixed, (5, 5), 0)
    laplacian_moving = cv2.Laplacian(blurred_moving, cv2.CV_64F)
    laplacian_fixed = cv2.Laplacian(blurred_fixed, cv2.CV_64F)
    # abs
    laplacian_fixed = np.abs(laplacian_fixed)
    laplacian_moving = np.abs(laplacian_moving)

    tx = multimodal_registration(laplacian_fixed, laplacian_moving)

    aligned = sitk.Resample(sitk.GetImageFromArray(moving), sitk.GetImageFromArray(fixed), tx, sitk.sitkLinear, 0.0, sitk.sitkUInt8)
    aligned = sitk.GetArrayFromImage(aligned)
    # make a composite of aligned and fixed, covert both to color and make aligned red
    fixed = cv2.cvtColor(fixed, cv2.COLOR_GRAY2BGR)
    aligned = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
    fixed[:, :, 2] = 0
    aligned[:, :, 0] = 0
    composite = cv2.addWeighted(fixed, 0.5, aligned, 0.5, 0)
    cv2.imshow("composite", composite)
    cv2.waitKey()
    cv2.destroyAllWindows()