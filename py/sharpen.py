import cv2
from pathlib import Path
import numpy as np
from skimage.filters import unsharp_mask
import argparse
import tifffile as tiff

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
            # Apply unsharp mask to enhance edges
            img = unsharp_mask(img, radius=radius, amount=amount)


            # convert to 8 bit tiff if not already
            if img.dtype != np.uint8:
                # if floating point
                if img.dtype == np.float32 or img.dtype == np.float64:
                    img = img * 255
                    img = img.astype(np.uint8)
                elif img.dtype == np.uint16:
                    img = (img / 256).astype(np.uint8)
                else:
                    raise Exception(f"Unsupported dtype: {img.dtype}")
        except Exception as e:
            print(f"Failed to process {file}. Error: {e}", flush=True)
            continue

        # Get filename stem
        stem = file.stem

        if args.equalize:
            # Equalize the histogram
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
            img = clahe.apply(img)
            # adjust brightness
            mask = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY)[1]
            enhanced = np.where(mask == 255, img + 30, img)
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            img = enhanced
        # Save the processed image
        cv2.imwrite(str(output_path / f"{stem}.tif"), img)

    print("Done!", flush=True)
