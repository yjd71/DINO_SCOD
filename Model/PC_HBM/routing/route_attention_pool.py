"""Attention pooling over the five camouflage-context descriptors."""

from __future__ import annotations

import torch
import torch.nn as nn

from ..common.utils import normalize


class RouteAttentionPool(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        hidden = max(16, int(dim) // 2)
        self.score = nn.Sequential(
            nn.Linear(int(dim), hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.ndim != 3 or tokens.size(1) != 5:
            raise ValueError(f"route tokens must be [B,5,D], got {tuple(tokens.shape)}")
        weights = torch.softmax(self.score(tokens).squeeze(-1), dim=1)
        pooled = (weights.unsqueeze(-1) * tokens).sum(dim=1)
        return normalize(pooled, dim=-1), weights


__all__ = ["RouteAttentionPool"]

