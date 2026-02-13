"""Training datasets for the slice position estimator.

Provides datasets for both synthetic atlas slices (with known ground truth)
and real tissue images (for domain adaptation). Supports configurable
preprocessing: CLAHE (default), Sobel (legacy), or none.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


def _sobel_edges(image: NDArray) -> NDArray:
    """Apply Sobel edge detection matching the original preprocessing (legacy)."""
    blurred = cv2.GaussianBlur(image, (3, 3), sigmaX=0, sigmaY=0)
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3, delta=25)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3, delta=25)
    gx = cv2.convertScaleAbs(gx)
    gy = cv2.convertScaleAbs(gy)
    return cv2.addWeighted(gx, 0.5, gy, 0.5, 0)


def _apply_preprocessing(image: NDArray, mode: str) -> NDArray:
    """Apply the selected preprocessing to an image.

    Args:
        image: Grayscale uint8 image.
        mode: One of "clahe", "sobel", or "none".

    Returns:
        Preprocessed uint8 image.
    """
    if mode == "clahe":
        from belljar.estimation.data_generation import clahe_normalize

        return clahe_normalize(image)
    elif mode == "sobel":
        return _sobel_edges(image)
    else:
        return image


class AngledAtlasDataset(Dataset):
    """Dataset of atlas slices with known positions and angles.

    Each sample is a 2D atlas slice rendered at a known (position, x_angle, y_angle),
    stored as PNG images with a metadata pickle file containing the labels.

    Supports both the legacy 3-output format and the new 9-output anchoring
    vector format. When metadata contains an 'anchoring' key, those values
    are used directly instead of converting from 3-value format.
    """

    def __init__(
        self,
        data_path: Path,
        transform: transforms.Compose | None = None,
        output_format: str = "anchoring",
        ap_range: tuple[float, float] = (0.0, 1324.0),
        angle_range: tuple[float, float] = (-10.0, 10.0),
        preprocessing: str = "clahe",
    ) -> None:
        self.data_path = data_path
        self.transform = transform
        self.output_format = output_format
        self.ap_range = ap_range
        self.angle_range = angle_range
        self.preprocessing = preprocessing

        self.file_list = sorted([
            f.name for f in data_path.iterdir()
            if f.is_file() and f.suffix == ".png"
        ])

        metadata_path = data_path / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                self.metadata: dict = pickle.load(f)
        else:
            self.metadata = {}
            logger.warning("No metadata.pkl found in %s", data_path)

    def __len__(self) -> int:
        return len(self.file_list)

    def _normalize_legacy(self, pos: float, x_angle: float, y_angle: float) -> torch.Tensor:
        """Normalize legacy 3-value labels to [0, 1]."""
        pos_norm = (pos - self.ap_range[0]) / (self.ap_range[1] - self.ap_range[0])
        x_norm = (x_angle - self.angle_range[0]) / (self.angle_range[1] - self.angle_range[0])
        y_norm = (y_angle - self.angle_range[0]) / (self.angle_range[1] - self.angle_range[0])
        return torch.tensor([pos_norm, x_norm, y_norm], dtype=torch.float32)

    def _to_anchoring(self, pos: float, x_angle: float, y_angle: float) -> torch.Tensor:
        """Convert position and angles to 9-value anchoring vectors."""
        from belljar.estimation.predictor import legacy_to_anchoring

        anchoring = legacy_to_anchoring(pos, x_angle, y_angle, self.ap_range)
        return torch.tensor(anchoring, dtype=torch.float32)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.item()

        filename = self.file_list[idx]
        img_path = self.data_path / filename

        image = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Failed to read image: {img_path}")

        image = _apply_preprocessing(image, self.preprocessing)

        if self.transform:
            image = self.transform(image)

        stem = Path(filename).stem
        label_data = self.metadata.get(stem, {})

        # New format: use precomputed anchoring vectors directly
        if self.output_format == "anchoring" and "anchoring" in label_data:
            anchoring = label_data["anchoring"]
            label = torch.tensor(anchoring, dtype=torch.float32)
        elif self.output_format == "anchoring":
            pos = float(label_data.get("pos", 0))
            x_angle = float(label_data.get("x_angle", 0))
            y_angle = float(label_data.get("y_angle", 0))
            label = self._to_anchoring(pos, x_angle, y_angle)
        else:
            pos = float(label_data.get("pos", 0))
            x_angle = float(label_data.get("x_angle", 0))
            y_angle = float(label_data.get("y_angle", 0))
            label = self._normalize_legacy(pos, x_angle, y_angle)

        return image, label


class TissueDataset(Dataset):
    """Dataset of real tissue images (unlabeled).

    Used for inference or domain evaluation. Supports configurable
    preprocessing.
    """

    def __init__(
        self,
        data_path: Path,
        transform: transforms.Compose | None = None,
        target_size: tuple[int, int] = (256, 256),
        preprocessing: str = "clahe",
    ) -> None:
        self.data_path = data_path
        self.transform = transform
        self.target_size = target_size
        self.preprocessing = preprocessing

        self.file_list = sorted([
            f.name for f in data_path.iterdir()
            if f.is_file() and f.suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
        ])

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> torch.Tensor:
        if torch.is_tensor(idx):
            idx = idx.item()

        img_path = self.data_path / self.file_list[idx]
        image = Image.open(img_path).convert("L")

        if image.size != self.target_size:
            image = image.resize(self.target_size)

        image_np = np.array(image)
        image_np = _apply_preprocessing(image_np, self.preprocessing)

        if self.transform:
            return self.transform(image_np)

        return transforms.ToTensor()(image_np)


class GaussianNoise:
    """Transform that adds Gaussian noise to a tensor."""

    def __init__(self, mean: float = 0.0, std: float = 1.0) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"
