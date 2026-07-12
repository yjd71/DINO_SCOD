"""Geometry compatibility scoring for parent-child hypotheses."""

from __future__ import annotations

import torch
import torch.nn as nn


class GeoScoreMLP(nn.Module):
    def __init__(self, geometry_dim: int = 6, hidden: int = 64) -> None:
        super().__init__()
        self.geometry_dim = int(geometry_dim)
        self.net = nn.Sequential(
            nn.Linear(self.geometry_dim * 3 + 3, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), 1),
        )

    def forward(
        self,
        parent_geo: torch.Tensor,
        child_geo: torch.Tensor,
        query_geo: torch.Tensor,
    ) -> torch.Tensor:
        if parent_geo.shape != child_geo.shape or parent_geo.ndim != 3:
            raise ValueError("parent_geo and child_geo must have matching [M,K,G] shapes")
        if query_geo.shape != (parent_geo.size(0), self.geometry_dim):
            raise ValueError(
                f"query_geo must be [M,{self.geometry_dim}], got {tuple(query_geo.shape)}"
            )
        query = query_geo.unsqueeze(1).expand_as(parent_geo)
        parent_child_delta = (parent_geo - child_geo).abs().mean(dim=-1, keepdim=True)
        parent_query_delta = (parent_geo - query).abs().mean(dim=-1, keepdim=True)
        child_query_delta = (child_geo - query).abs().mean(dim=-1, keepdim=True)
        features = torch.cat(
            (
                parent_geo,
                child_geo,
                query,
                parent_child_delta,
                parent_query_delta,
                child_query_delta,
            ),
            dim=-1,
        )
        return self.net(features).squeeze(-1)


__all__ = ["GeoScoreMLP"]

