"""P2-only local child-query construction for PC-HBM-Lite."""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from ..common.utils import gather_local_patches
from .child_local_encoder import ChildLocalEncoder


class ChildQueryBuilder(nn.Module):
    """Encode an odd-window P2 patch at every selected P3 coordinate."""

    def __init__(self, p2_ch: int, dim: int = 128, window: int = 3) -> None:
        super().__init__()
        self.p2_ch = int(p2_ch)
        self.dim = int(dim)
        self.window = int(window)
        if self.p2_ch <= 0:
            raise ValueError("p2_ch must be positive")
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.window <= 0 or self.window % 2 == 0:
            raise ValueError("window must be a positive odd integer")
        self.encoder = ChildLocalEncoder(
            self.p2_ch, dim=self.dim, window=self.window
        )

    def encode_child_map(
        self,
        p2: torch.Tensor,
        batch_ids: torch.Tensor,
        flat_indices: torch.Tensor,
        p3_hw: Tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        if p2.ndim != 4 or p2.size(1) != self.p2_ch:
            raise ValueError(
                f"p2 must be [B,{self.p2_ch},H,W], got {tuple(p2.shape)}"
            )
        p2_hw = tuple(int(value) for value in p2.shape[-2:])
        if p3_hw is not None and tuple(int(value) for value in p3_hw) != p2_hw:
            raise ValueError(
                "PC-HBM-Lite requires aligned P2/P3 grids for Pair Memory"
            )
        patches = gather_local_patches(
            p2,
            batch_ids,
            flat_indices,
            window=self.window,
        )
        return {
            "q_child": self.encoder(patches),
            "child_patches": patches,
            "flat_indices2_from_p3": flat_indices,
        }

    def forward(
        self,
        p2: torch.Tensor,
        batch_ids: torch.Tensor,
        flat_indices: torch.Tensor,
        p3_hw: Tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.encode_child_map(p2, batch_ids, flat_indices, p3_hw)
