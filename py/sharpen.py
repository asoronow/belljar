import cv2
from pathlib import Path
import numpy as np
from skimage.filters import unsharp_mask
from skimage.morphology import white_tophat, disk
import argparse
import tifffile as tiff


def enhance_contrast(image, saturation_level=0.05):
    """
    Enhance the contrast of an image by saturating a certain percentage of pixels,
    agnostic of the image's data type.

    Parameters:
    - image: NumPy array of the image.
    - saturation_level: Percentage of pixels to saturate at both low and high ends of the intensity spectrum.

    Returns:
    - The image with enhanced contrast.
    """
    # Ensure the saturation level is expressed as a fraction.
    saturation_point = saturation_level / 100

    # Flatten the image array to work with intensity values linearly.
    flat_image = image.ravel()

    # Determine the intensity values at the low and high saturation points.
    low_saturation_value = np.percentile(flat_image, saturation_point)
    high_saturation_value = np.percentile(flat_image, 100 - saturation_point)

    # Clip the intensity values to the determined range.
    clipped_image = np.clip(flat_image, low_saturation_value, high_saturation_value)

    # Dynamically determine the min and max intensity values based on the data type of the input image.
    dtype_min, dtype_max = np.iinfo(image.dtype).min, np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else (np.finfo(image.dtype).min, np.finfo(image.dtype).max)

    # Rescale the intensity values to span the full range of the data type.
    rescaled_image = np.interp(clipped_image, (clipped_image.min(), clipped_image.max()), (dtype_min, dtype_max))

    # Reshape the flat array back to the original image shape.
    enhanced_image = rescaled_image.reshape(image.shape)

    # Ensure the enhanced image has the same data type as the input image.
    return enhanced_image.astype(image.dtype)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process z-stack images")
    parser.add_argument(
        "-o",
        "--output",
        help="output directory",
        default="",
    )
    parser.add_argument("-i", "--input", help="input directory", default="")
    parser.add_argument("-r", "--radius", help="radius for unsharp mask", default=3)
    parser.add_argument("-a", "--amount", help="amount for unsharp mask", default=2)
    parser.add_argument(
        "-e",
        "--equalize",
        help="equalize histogram",
        action="store_true",
    )

    args = parser.parse_args()

    input_path = Path(args.input.strip())
    output_path = Path(args.output.strip())
    amount = float(args.amount.strip())
    radius = float(args.radius.strip())
    valid_extensions = [".tif", ".tiff"]
    input_files = [
        file for file in input_path.iterdir() if file.suffix in valid_extensions
    ]
    input_files.sort()
    print(f"{len(input_files)}", flush=True)
    for file in input_files:
        try:
            print(f"Processing {file}", flush=True)
            img = tiff.imread(file)
            if args.equalize:
                # Perform contrast enhancement
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
                img = clahe.apply(img)
                img = enhance_contrast(img)
            
            original_dtype = img.dtype
            # Apply unsharp mask to enhance edges
            img = unsharp_mask(img, radius=radius, amount=amount, preserve_range=True)
            img = white_tophat(img, disk(15))
            # Convert the image back to its original data type
            img = img.astype(original_dtype)
   
        except Exception as e:
            print(f"Failed to process {file}. Error: {e}", flush=True)
            continue

        # Get filename stem
        stem = file.stem
        # Save the processed image
        cv2.imwrite(f"{output_path}/{stem}.png", img)

    print("Done!", flush=True)
