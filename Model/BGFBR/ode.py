"""Orthogonal dual enhancement block adapted from FEDER."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualFunction(nn.Sequential):
    def __init__(self, channels: int) -> None:
        super().__init__(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )


class ODEBlock(nn.Module):
    """Blend two residual transformations with a learned per-image scalar."""

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = int(channels)
        self.f1 = _ResidualFunction(channels)
        self.f2 = _ResidualFunction(channels)
        self.alpha_mlp = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

    def forward(
        self,
        feature: torch.Tensor,
        *,
        return_alpha: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if feature.ndim != 4 or feature.shape[1] != self.channels:
            raise ValueError(
                f"ODE input must have shape [B,{self.channels},H,W], got {tuple(feature.shape)}"
            )
        f1 = self.f1(feature)
        f2 = self.f2(f1 + feature)
        joint = torch.cat((f1, f2), dim=1)
        pooled = F.adaptive_avg_pool2d(joint, 1) + F.adaptive_max_pool2d(joint, 1)
        alpha = torch.sigmoid(self.alpha_mlp(pooled))
        output = feature + alpha * f1 + (1.0 - alpha) * f2
        if return_alpha:
            return output, alpha
        return output


OrthogonalDualEnhancement = ODEBlock


__all__ = ["ODEBlock", "OrthogonalDualEnhancement"]
