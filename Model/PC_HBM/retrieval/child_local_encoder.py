"""Configurable odd-window P2 local-key encoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ChildLocalEncoder(nn.Module):
    def __init__(self, in_ch: int, dim: int = 128, window: int = 3) -> None:
        super().__init__()
        self.window = int(window)
        self.dim = int(dim)
        if self.window <= 0 or self.window % 2 == 0:
            raise ValueError("window must be a positive odd integer")
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        hidden = max(64, self.dim // 2)
        groups = 8 if hidden % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(int(in_ch), hidden, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, self.dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 4:
            raise ValueError(
                "patches must be [M,C,window,window], "
                f"got {tuple(patches.shape)}"
            )
        if patches.shape[-2:] != (self.window, self.window):
            raise ValueError(
                f"Expected {self.window}x{self.window} patches, "
                f"got {tuple(patches.shape[-2:])}"
            )
        if patches.size(0) == 0:
            return patches.new_empty((0, self.dim))
        encoded = self.net(patches).flatten(1)
        projected = torch.nan_to_num(
            self.proj(encoded).float(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        normalized = F.normalize(projected, dim=-1, eps=1.0e-6)
        return normalized.to(dtype=patches.dtype)
