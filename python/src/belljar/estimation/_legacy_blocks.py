"""Legacy ResNet building blocks for backward compatibility with v1 weights.

These reproduce the exact architecture from model.py so that
existing trained weights can be loaded.
"""

from __future__ import annotations

import torch.nn as nn


class ResidualBlock(nn.Module):
    """Bottleneck residual block matching the v1 TissuePredictor architecture."""

    def __init__(
        self,
        in_channels: int,
        intermediate_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, intermediate_channels, kernel_size=1, stride=stride, padding=0
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels, intermediate_channels,
            kernel_size=3, stride=1, padding=1,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut: nn.Module = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":  # noqa: F821
        import torch

        residual = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += residual
        return self.relu(out)
