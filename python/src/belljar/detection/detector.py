"""Cell/neuron detection using YOLO + SAHI.

Refactored from find_neurons.py with configurable parameters
and proper type hints.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff
import torch
from numpy.typing import NDArray
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from skimage.exposure import equalize_adapthist

from belljar.config import DetectionConfig
from belljar.detection.screening import screen_predictions
from belljar.types import DetectionResult

logger = logging.getLogger(__name__)


def _normalize_image(img: NDArray) -> NDArray:
    """Normalize image to uint8 range regardless of input dtype."""
    if img.dtype == np.uint16:
        return (img / 256).astype(np.uint8)
    elif img.dtype in (np.float32, np.float64):
        return (img * 255).astype(np.uint8)
    return img.astype(np.uint8)


def _prepare_for_detection(img: NDArray) -> NDArray:
    """Apply CLAHE and convert to BGR for YOLO input."""
    img = _normalize_image(img)
    img = equalize_adapthist(img, clip_limit=0.01)
    img = (img * 255).astype(np.uint8)
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def load_image(file_path: Path) -> tuple[NDArray, str]:
    """Load an image file, handling TIFF and standard formats.

    Returns:
        Tuple of (image_array, index_order) where index_order is
        "C" for channel-first TIFFs and "F" for channel-last.
    """
    ext = file_path.suffix.lower()
    if ext in (".tif", ".tiff"):
        img = tiff.imread(str(file_path))
        if len(img.shape) == 3:
            return img, "C"
        return img, "F"
    else:
        img = cv2.imread(str(file_path))
        if img is None:
            raise ValueError(f"Failed to read image: {file_path}")
        return img, "F"


def detect_neurons(
    image: NDArray,
    model: AutoDetectionModel,
    config: DetectionConfig,
    channel_index: int = 0,
) -> DetectionResult:
    """Run neuron detection on a single image/channel.

    Args:
        image: Input image (grayscale or BGR uint8).
        model: Pre-loaded SAHI detection model.
        config: Detection configuration.
        channel_index: Channel index for multi-channel images.

    Returns:
        DetectionResult with bounding boxes and scores.
    """
    prepared = _prepare_for_detection(image)

    result = get_sliced_prediction(
        prepared,
        model,
        slice_height=config.tile_size,
        slice_width=config.tile_size,
        overlap_height_ratio=config.overlap_ratio,
        overlap_width_ratio=config.overlap_ratio,
    )

    filtered = screen_predictions(
        result.object_prediction_list,
        area_threshold=config.area_threshold,
        eccentricity_threshold=config.eccentricity_threshold,
        image=prepared,
    )

    boxes = [obj.bbox.to_xyxy() for obj in filtered]
    scores = [obj.score.value for obj in filtered]

    height, width = image.shape[:2]
    return DetectionResult(
        boxes=boxes,
        scores=scores,
        image_width=width,
        image_height=height,
        channel_index=channel_index,
        model_name=config.model_name,
        confidence_threshold=config.confidence_threshold,
    )


def create_detection_model(
    model_path: str | Path,
    config: DetectionConfig,
) -> AutoDetectionModel:
    """Create and configure a SAHI detection model.

    Args:
        model_path: Path to the YOLO model weights.
        config: Detection configuration.

    Returns:
        Configured AutoDetectionModel ready for inference.
    """
    if torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"

    logger.info("Using device: %s, model: %s", device, model_path)

    return AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=str(model_path),
        confidence_threshold=config.confidence_threshold,
        device=device,
    )


def export_bboxes(image: NDArray, boxes: list[list[float]], output_path: Path) -> None:
    """Draw bounding boxes on an image and save to disk.

    Args:
        image: Input image (BGR uint8).
        boxes: List of [x1, y1, x2, y2] bounding boxes.
        output_path: Path to save the annotated image.
    """
    result = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = [int(b) for b in box]
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imwrite(str(output_path), result)
