"""Build per-boundary-query states for parent-child cross-attention."""

from __future__ import annotations

import torch
from torch import nn

from ..common.utils import normalize


class QueryStateBuilder(nn.Module):
    """Fuse p3, child, route and uncertainty evidence into ``[M,D]``."""

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        self.dim = int(dim)
        self.net = nn.Sequential(
            nn.Linear(self.dim * 3 + 2, self.dim),
            nn.GELU(),
            nn.Linear(self.dim, self.dim),
        )

    def forward(
        self,
        q3: torch.Tensor,
        q_child: torch.Tensor,
        route_context_token: torch.Tensor,
        c23: torch.Tensor,
        parent_entropy: torch.Tensor,
        query_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if q3.ndim != 2 or q3.size(1) != self.dim:
            raise ValueError(f"q3 must be [M,{self.dim}], got {tuple(q3.shape)}")
        if q_child.shape != q3.shape or route_context_token.shape != q3.shape:
            raise ValueError("q_child and route_context_token must match q3")
        if c23.ndim == 1:
            c23 = c23[:, None]
        if parent_entropy.ndim == 1:
            parent_entropy = parent_entropy[:, None]
        if c23.shape != (q3.size(0), 1) or parent_entropy.shape != (
            q3.size(0),
            1,
        ):
            raise ValueError("c23 and parent_entropy must be [M] or [M,1]")
        features = torch.cat(
            [q3, q_child, route_context_token, c23, parent_entropy], dim=-1
        )
        state = normalize(self.net(features), dim=-1)
        if query_valid is not None:
            valid = query_valid.to(device=state.device, dtype=state.dtype)
            if tuple(valid.shape) != (state.size(0),):
                raise ValueError("query_valid must be [M]")
            state = state * valid[:, None]
        return state

