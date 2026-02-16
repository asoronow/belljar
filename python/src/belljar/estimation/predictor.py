"""Slice position estimator using pretrained ResNet50 backbone.

Upgraded from the custom ResNet-101 in model.py:
- Pretrained ImageNet backbone (ResNet50) for better feature extraction
- 9 output values (3 anchoring points x 3D coordinates) for QuickNII/QUINT
  interoperability, following the DeepSlice approach (Nature Comms 2023)
- MC Dropout for uncertainty estimation
- Configurable via EstimationConfig

The 9 outputs represent three 3D points that define the cutting plane:
  [ox, oy, oz, ux, uy, uz, vx, vy, vz]
  - o = origin point of the section in atlas space
  - u = unit vector along the section width
  - v = unit vector along the section height

These can be converted back to (z_position, x_angle, y_angle) for
backward compatibility with v1 alignment format.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.typing import NDArray
from torchvision import models, transforms

from belljar.config import EstimationConfig

logger = logging.getLogger(__name__)


class SliceEstimator(nn.Module):
    """Pretrained ResNet50 backbone for slice position estimation.

    Replaces the custom ResNet-101 TissuePredictor with a pretrained backbone.
    Uses MC Dropout for uncertainty estimation during inference.
    """

    def __init__(self, num_outputs: int = 9, dropout_rate: float = 0.2) -> None:
        super().__init__()
        # Load pretrained ResNet50 (or use default weights=None for training from scratch)
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Replace first conv to accept 1-channel grayscale input
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Initialize from the pretrained 3-channel weights (average across channels)
        with torch.no_grad():
            self.conv1.weight.copy_(backbone.conv1.weight.mean(dim=1, keepdim=True))

        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        # Dropout for MC Dropout uncertainty estimation
        self.dropout = nn.Dropout(p=dropout_rate)

        # Prediction head: 2048 -> num_outputs
        self.fc = nn.Linear(2048, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature vector before the prediction head."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)


class LegacySliceEstimator(nn.Module):
    """Backward-compatible wrapper for v1 TissuePredictor weights.

    Loads the original 3-output ResNet-101 model and wraps predictions
    in the v2 interface (returning z_pos, x_angle, y_angle).
    """

    def __init__(self) -> None:
        super().__init__()
        # Replicate the original architecture
        from belljar.estimation._legacy_blocks import ResidualBlock

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv2 = self._make_layer(64, 64, 256, blocks=3, stride=1)
        self.conv3 = self._make_layer(256, 128, 512, blocks=4, stride=2)
        self.conv4 = self._make_layer(512, 256, 1024, blocks=23, stride=2)
        self.conv5 = self._make_layer(1024, 512, 2048, blocks=3, stride=2)
        self.fc = nn.Linear(2048, 3)

    def _make_layer(
        self, in_ch: int, mid_ch: int, out_ch: int, blocks: int, stride: int
    ) -> nn.Sequential:
        from belljar.estimation._legacy_blocks import ResidualBlock

        layers = [ResidualBlock(in_ch, mid_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_ch, mid_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.fc(x)


def _get_device() -> torch.device:
    """Determine the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def _get_transforms(config: EstimationConfig) -> transforms.Compose:
    """Build the image transforms for inference."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[config.normalization_mean], std=[config.normalization_std]),
    ])


def _preprocess_image(
    image: NDArray, config: EstimationConfig
) -> torch.Tensor:
    """Preprocess an image for the estimator model.

    Applies the configured preprocessing (CLAHE, Sobel, or none) followed
    by normalization.
    """
    # Ensure grayscale uint8
    if image.dtype != np.uint8:
        if image.dtype == np.uint16:
            image = (image / 256).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize
    image = cv2.resize(image, (config.input_size, config.input_size))

    # Apply configured preprocessing
    mode = config.preprocessing
    if mode == "clahe":
        from belljar.estimation.data_generation import clahe_normalize

        image = clahe_normalize(image)
    elif mode == "sobel":
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3, delta=25)
        gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3, delta=25)
        gx = cv2.convertScaleAbs(gx)
        gy = cv2.convertScaleAbs(gy)
        image = cv2.addWeighted(gx, 0.5, gy, 0.5, 0)
    # else: "none" — use raw grayscale

    # Apply transforms
    tx = _get_transforms(config)
    return tx(image).unsqueeze(0)  # Add batch dim


def predict_slice_position(
    image: NDArray,
    model: nn.Module,
    config: EstimationConfig,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Predict the atlas position for a single tissue section.

    Args:
        image: Tissue image (grayscale or BGR).
        model: Loaded estimator model.
        config: Estimation configuration.
        device: Device to run inference on.

    Returns:
        Dict with 'prediction' (raw model output), 'z_position', 'x_angle',
        'y_angle', and optionally 'uncertainty' if MC dropout is used.
    """
    if device is None:
        device = _get_device()

    input_tensor = _preprocess_image(image, config).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor).cpu().numpy()[0]

    result: dict[str, Any] = {"raw_output": output.tolist()}

    if len(output) == 9:
        # V2 anchoring vector format
        result["anchoring_vectors"] = {
            "ox": output[0], "oy": output[1], "oz": output[2],
            "ux": output[3], "uy": output[4], "uz": output[5],
            "vx": output[6], "vy": output[7], "vz": output[8],
        }
        # Convert to legacy format for backward compatibility
        z_pos, x_angle, y_angle = anchoring_to_legacy(output)
        result["z_position"] = z_pos
        result["x_angle"] = x_angle
        result["y_angle"] = y_angle
    elif len(output) == 3:
        # Legacy 3-value format (normalized)
        result["z_position"] = output[0]
        result["x_angle"] = output[1]
        result["y_angle"] = output[2]
    else:
        result["z_position"] = output[0] if len(output) > 0 else 0.0
        result["x_angle"] = output[1] if len(output) > 1 else 0.0
        result["y_angle"] = output[2] if len(output) > 2 else 0.0

    return result


def predict_with_uncertainty(
    image: NDArray,
    model: nn.Module,
    config: EstimationConfig,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Predict with MC Dropout uncertainty estimation.

    Runs N forward passes with dropout enabled to estimate prediction
    uncertainty. High variance indicates the model is uncertain.

    Args:
        image: Tissue image.
        model: Loaded estimator model with dropout layers.
        config: Estimation configuration.
        device: Device for inference.

    Returns:
        Dict with 'prediction' (mean), 'uncertainty' (std), and 'samples'.
    """
    if device is None:
        device = _get_device()

    input_tensor = _preprocess_image(image, config).to(device)
    n_samples = config.mc_dropout_samples

    # Enable dropout during inference
    model.train()
    samples = []

    with torch.no_grad():
        for _ in range(n_samples):
            output = model(input_tensor).cpu().numpy()[0]
            samples.append(output)

    model.eval()

    samples_array = np.array(samples)
    mean_pred = samples_array.mean(axis=0)
    std_pred = samples_array.std(axis=0)

    result = predict_slice_position(image, model, config, device)
    result["uncertainty"] = std_pred.tolist()
    result["mc_samples"] = samples_array.tolist()
    result["mean_prediction"] = mean_pred.tolist()

    return result


def anchoring_to_legacy(
    anchoring: NDArray | list[float],
    ap_range: tuple[float, float] = (0.0, 1324.0),
    angle_range: tuple[float, float] = (-10.0, 10.0),
) -> tuple[float, float, float]:
    """Convert 9-value anchoring vectors to legacy (z_pos, x_angle, y_angle).

    The anchoring vector defines the cutting plane via three 3D points.
    We extract the AP position from the origin's z-coordinate and compute
    the tilt angles from the plane normal.

    Args:
        anchoring: Array of 9 values [ox,oy,oz, ux,uy,uz, vx,vy,vz].
        ap_range: Min/max AP position for denormalization.
        angle_range: Min/max angle for denormalization.

    Returns:
        Tuple of (z_position, x_angle, y_angle) in physical units.
    """
    a = np.array(anchoring, dtype=np.float64)
    o = a[0:3]  # Origin
    u = a[3:6]  # Width vector
    v = a[6:9]  # Height vector

    # AP position from origin z-coordinate
    z_pos = o[2] * (ap_range[1] - ap_range[0]) + ap_range[0]

    # Plane normal from cross product of u and v
    normal = np.cross(u, v)
    norm = np.linalg.norm(normal)
    if norm > 1e-8:
        normal = normal / norm

    # Tilt angles from normal vector
    # x_angle: tilt around x-axis (related to normal's y component)
    # y_angle: tilt around y-axis (related to normal's x component)
    x_angle = np.degrees(np.arctan2(normal[1], normal[2]))
    y_angle = np.degrees(np.arctan2(normal[0], normal[2]))

    return float(z_pos), float(x_angle), float(y_angle)


def legacy_to_anchoring(
    z_position: float,
    x_angle: float,
    y_angle: float,
    ap_range: tuple[float, float] = (0.0, 1324.0),
) -> list[float]:
    """Convert legacy (z_pos, x_angle, y_angle) to 9-value anchoring vectors.

    Args:
        z_position: AP position in atlas coordinates.
        x_angle: X tilt angle in degrees.
        y_angle: Y tilt angle in degrees.
        ap_range: Min/max AP position for normalization.

    Returns:
        List of 9 values [ox,oy,oz, ux,uy,uz, vx,vy,vz].
    """
    # Normalize z_position
    z_norm = (z_position - ap_range[0]) / (ap_range[1] - ap_range[0])

    # Origin at center of the section plane
    o = [0.5, 0.5, z_norm]

    # Compute rotation from angles
    x_rad = np.radians(x_angle)
    y_rad = np.radians(y_angle)

    # Width vector (initially along x-axis, rotated by y_angle)
    u = [np.cos(y_rad), 0.0, -np.sin(y_rad)]

    # Height vector (initially along y-axis, rotated by x_angle)
    v = [0.0, np.cos(x_rad), -np.sin(x_rad)]

    return o + u + v


def load_model(
    model_path: Path,
    config: EstimationConfig,
    device: torch.device | None = None,
    legacy: bool = False,
) -> nn.Module:
    """Load a slice estimator model from disk.

    Args:
        model_path: Path to the .pt weights file.
        config: Estimation configuration.
        device: Device to load model onto.
        legacy: If True, load the v1 TissuePredictor architecture.

    Returns:
        Loaded model in eval mode.
    """
    if device is None:
        device = _get_device()

    if legacy:
        model = LegacySliceEstimator()
    else:
        model = SliceEstimator(num_outputs=9)

    checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)

    # Support enriched checkpoint format from train.py
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        logger.info(
            "Loaded enriched checkpoint (epoch=%d, val_loss=%.6f)",
            checkpoint.get("epoch", -1),
            checkpoint.get("val_loss", float("nan")),
        )
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    logger.info("Loaded estimator model from %s (legacy=%s)", model_path, legacy)
    return model
