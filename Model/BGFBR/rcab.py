"""Residual channel-attention block used by BGFBR refinement branches."""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int = 128, reduction: int = 16) -> None:
        super().__init__()
        if channels <= 0 or reduction <= 0:
            raise ValueError("channels and reduction must be positive")
        hidden = max(channels // reduction, 1)
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature * self.body(feature)


class RCAB(nn.Module):
    """Conv-ReLU-Conv-channel-attention residual body."""

    def __init__(self, channels: int = 128, reduction: int = 16) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            ChannelAttention(channels, reduction),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return feature + self.body(feature)


ResidualChannelAttentionBlock = RCAB


__all__ = ["ChannelAttention", "RCAB", "ResidualChannelAttentionBlock"]
