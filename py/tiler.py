import cv2
import numpy as np
import pickle
from pathlib import Path
from find_neurons import DetectionResult
import argparse


class ImageTile:
    """
    Represents a single tile from an image.

    Parameters
    ----------
    image : np.ndarray
        The image of the tile.
    x_center : int
        The x center of the tile in the original image.
    y_center : int
        The y center of the tile in the original image.
    width : int
        The width of the tile.
    height : int
        The height of the tile.
    """

    def __init__(self, image, x_center, y_center, width, height):
        self.image = image
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height

    def bbox_in_tile(self, bbox):
        """
        Convert a bbox to be relative to the tile.

        Parameters
        ----------
        bbox : list
            The bbox to convert.

        Returns
        -------
        list
            The converted bbox.
        """
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        x1 -= self.x_center
        y1 -= self.y_center
        x2 -= self.x_center
        y2 -= self.y_center

        return [x1, y1, x2, y2]

    def bbox_to_yolo(self, bbox):
        """
        Convert a bbox to YOLO format normalized xywh

        Parameters
        ----------
        bbox : list
            The bbox to convert.

        Returns
        -------
        list
            The converted bbox.
        """
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        x_center /= self.width
        y_center /= self.height
        w /= self.width
        h /= self.height

        return [x_center, y_center, w, h]


def tile_image(image, tile_size, overlap=0):
    """
    Tile an image into smaller images.

    Parameters
    ----------
    image : np.ndarray
        The image to tile.
    tile_size : int
        The size of the tiles.
    overlap : int, optional
        The amount of overlap between tiles. Defaults to 0.
    Returns
    -------
    list
        A list of ImageTile objects.
    """
    tiles = []
    height, width = image.shape[:2]

    # Adjusted calculation for the number of tiles considering the overlap
    stride = tile_size - overlap
    # Create the tiles
    for y in range(0, height - overlap, stride):
        for x in range(0, width - overlap, stride):
            tile = image[y : y + tile_size, x : x + tile_size]
            x_center = x  + tile_size // 2
            y_center = y  + tile_size // 2
            tiles.append(ImageTile(tile, x_center, y_center, tile_size, tile_size))

    return tiles


def write_tiles_labels(tiles, bboxes, output_dir):
    """
    Write the tiles and the YOLO label of all bbox in the tiles.

    Parameters
    ----------
    tiles : list
        A list of ImageTile objects.
    bboxes : list
        A list of bboxes.
    output_dir : str
        The output directory.
    """
    # Path
    output_dir = Path(output_dir).expanduser()

    # Write the tiles and the YOLO label of all bbox in the tiles
    for i, tile in enumerate(tiles):
        # Write the tile
        tile_path = output_dir / f"tile_{i}.png"

        # Check what bboxes are in the tile
        tile_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            if x1 >= tile.x_center and x2 <= tile.x_center + tile.width:
                if y1 >= tile.y_center and y2 <= tile.y_center + tile.height:
                    tile_bboxes.append(tile.bbox_in_tile(bbox))

        if len(tile_bboxes) > 0:
            cv2.imwrite(str(tile_path), tile.image)
            # Convert to YOLO format
            yolo_bboxes = [tile.bbox_to_yolo(bbox) for bbox in tile_bboxes]

            # Show boxes on image
            for bbox in tile_bboxes:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(tile.image, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # equalize the image
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            tile.image = clahe.apply(tile.image)
            cv2.imshow("Tile", tile.image)
            cv2.waitKey(0)

            # Write the YOLO label
            label_path = output_dir / f"tile_{i}.txt"
            with open(label_path, "w") as f:
                for bbox in yolo_bboxes:
                    # assuming class is 0
                    f.write(f"0 {' '.join(str(coord) for coord in bbox)}\n")
        else:
            continue


def load_predictions(input_dir):
    """
    Loads predictions from Bell Jar cell detection model.

    Parameters
    ----------
    input_dir : str
        The input directory.

    Returns
    -------
    list
        A list of bbox predictions.
    """
    # Path
    input_dir = Path(input_dir).expanduser()

    # Load the predictions
    predictions = list(input_dir.glob("*.pkl"))
    bboxes = []
    for prediction in predictions:
        with open(prediction, "rb") as f:
            fdata = pickle.load(f)
            bboxes += fdata[0].boxes

    return bboxes


def pull_images(input_dir, output_dir):
    """
    Pulls and tiles images from the input directory.

    Parameters
    ----------
    input_dir : str
        The input directory.
    output_dir : str
        The output directory.
    bboxes : list
        A list of bboxes.
    """
    # Path
    input_dir = Path(input_dir).expanduser()
    output_dir = Path(output_dir).expanduser()

    # Check the input dir for any sub directories calld "Max Projection"
    max_projection_dirs = list(input_dir.rglob("*/Max Projection"))
    # Get all the tifs in these directories
    images = []
    for max_projection_dir in max_projection_dirs:
        images += list(max_projection_dir.rglob("*.tiff"))
    
    print(f"Found {len(images)} images")
    # make 640x640 tiles of every image
    tile_num = 0
    for image in images:
        # Load the image
        image = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        # convert to 8 bit
        image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

        # Tile the image
        tiles = []
        width, height = image.shape
        for y in range(0, height, 640):
            for x in range(0, width, 640):
                tile = image[y : y + 640, x : x + 640]
                tiles.append(tile)
        print(f"Made {len(tiles)} tiles")
        # Write the tiles to the output directory
        for tile in tiles:
            tile_path = output_dir / f"tile_{tile_num}.png"
            tile_num += 1
            if np.mean(tile) > 10:
                cv2.imwrite(str(tile_path), tile)


def main():
    # Parse the arguments
    parser = argparse.ArgumentParser(description="Tile an image and write YOLO labels.")
    parser.add_argument(
        "images_path", type=str, help="The path to the images to tile and label."
    )
    parser.add_argument(
        "detection_dir",
        type=str,
        help="The directory containing the predictions from the cell detection model.",
    )
    parser.add_argument(
        "output_dir", type=str, help="The directory to write the tiles and labels."
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=320,
        help="The size of the tiles. Defaults to 320.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=0,
        help="The amount of overlap between tiles. Defaults to 0.",
    )
    args = parser.parse_args()

    # Find all the tifs
    images_path = Path(args.images_path).expanduser()
    images = list(images_path.rglob("*.tif"))
    # Load the predictions
    pull_images(args.images_path, args.output_dir)


if __name__ == "__main__":
    main()
