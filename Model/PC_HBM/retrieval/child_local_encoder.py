"""Encode 5x5 local child patches into normalized 128-D keys."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..common.utils import normalize


class ChildLocalEncoder(nn.Module):
    def __init__(self, in_ch: int, dim: int = 128, window: int = 5) -> None:
        super().__init__()
        self.window = int(window)
        hidden = max(64, int(dim) // 2)
        groups = 8 if hidden % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(int(in_ch), hidden, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden),
            nn.GELU(),
            nn.Conv2d(hidden, int(dim), kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(int(dim), int(dim))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.ndim != 4:
            raise ValueError(f"patches must be [M,C,H,W], got {tuple(patches.shape)}")
        if patches.shape[-2:] != (self.window, self.window):
            raise ValueError(
                f"Expected {self.window}x{self.window} patches, got {tuple(patches.shape[-2:])}"
            )
        if patches.size(0) == 0:
            return patches.new_empty((0, self.proj.out_features))
        encoded = self.net(patches).flatten(1)
        return normalize(self.proj(encoded), dim=-1)


__all__ = ["ChildLocalEncoder"]

