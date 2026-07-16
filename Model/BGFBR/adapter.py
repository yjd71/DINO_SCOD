"""Residual adapter for the deepest projected DINO feature."""

from __future__ import annotations

import torch
import torch.nn as nn


class F4Adapter(nn.Module):
    """Zero-initialized residual ``C -> bottleneck -> C`` adapter."""

    def __init__(self, channels: int = 128, bottleneck: int = 32, gamma: float = 1.0) -> None:
        super().__init__()
        if channels <= 0 or bottleneck <= 0:
            raise ValueError("channels and bottleneck must be positive")
        self.down = nn.Conv2d(channels, bottleneck, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)
        self.up = nn.Conv2d(bottleneck, channels, kernel_size=1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.gamma = nn.Parameter(torch.tensor(float(gamma), dtype=torch.float32))

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        residual = self.up(self.activation(self.down(feature)))
        return feature + self.gamma.to(dtype=feature.dtype) * residual


__all__ = ["F4Adapter"]
