"""Structured foreground/background prior for child verification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StructuredPriorBiasNet(nn.Module):
    def __init__(self, value_dim: int = 8, geometry_dim: int = 6, hidden: int = 64) -> None:
        super().__init__()
        input_dim = int(value_dim) + int(geometry_dim) * 2 + 4
        self.residual = nn.Sequential(
            nn.Linear(input_dim, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), 1),
        )
        self.gamma_prior = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        parent_values: torch.Tensor,
        parent_geo: torch.Tensor,
        child_geo: torch.Tensor,
        child_scores: torch.Tensor,
        geometry_scores: torch.Tensor,
    ) -> torch.Tensor:
        parent_fg = parent_values[..., 4]
        parent_bg = parent_values[..., 5]
        child_positive = torch.sigmoid(child_scores)
        geometry_positive = torch.sigmoid(geometry_scores)
        contradiction = parent_bg * child_positive + parent_fg * (1.0 - child_positive)
        prior_base = (
            F.softplus(parent_fg)
            + F.softplus(child_positive)
            + F.softplus(geometry_positive)
            - F.softplus(contradiction)
        )
        geometry_delta = (parent_geo - child_geo).abs().mean(dim=-1, keepdim=True)
        features = torch.cat(
            (
                parent_values,
                parent_geo,
                child_geo,
                child_scores.unsqueeze(-1),
                geometry_scores.unsqueeze(-1),
                geometry_delta,
                contradiction.unsqueeze(-1),
            ),
            dim=-1,
        )
        residual = self.residual(features).squeeze(-1)
        return prior_base + self.gamma_prior.tanh() * residual


__all__ = ["StructuredPriorBiasNet"]

