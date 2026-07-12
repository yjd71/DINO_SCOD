"""Prediction-structured child queries gathered with image-wise ``F.unfold``."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.utils import (
    gather_local_patches,
    gather_tokens,
    geometry_map_from_logits,
    scale_flat_indices,
)
from .child_local_encoder import ChildLocalEncoder


class DinoChildQueryBuilder(nn.Module):
    def __init__(self, p2_ch: int, dim: int = 128, window: int = 5) -> None:
        super().__init__()
        self.window = int(window)
        self.dim = int(dim)
        self.encoder = ChildLocalEncoder(int(p2_ch), dim=self.dim, window=self.window)
        self.geo_residual = nn.Sequential(
            nn.Linear(self.dim + 6, 64),
            nn.GELU(),
            nn.Linear(64, 6),
        )
        nn.init.zeros_(self.geo_residual[-1].weight)
        nn.init.zeros_(self.geo_residual[-1].bias)

    def encode_child_map(
        self,
        child_map: torch.Tensor,
        batch_ids3: torch.Tensor,
        flat_indices3: torch.Tensor,
        p3_hw: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        """Encode child memory keys without inventing query geometry."""

        if child_map.ndim != 4:
            raise ValueError(f"child_map must be [B,C,H,W], got {tuple(child_map.shape)}")
        p2_hw = tuple(int(value) for value in child_map.shape[-2:])
        flat_indices2 = scale_flat_indices(flat_indices3, p3_hw, p2_hw)
        patches = gather_local_patches(
            child_map,
            batch_ids3,
            flat_indices2,
            window=self.window,
        )
        return {
            "q_child": self.encoder(patches),
            "child_patches": patches,
            "flat_indices2_from_p3": flat_indices2,
        }

    def forward(
        self,
        child_map: torch.Tensor,
        m2_pre_logits: torch.Tensor,
        batch_ids3: torch.Tensor,
        flat_indices3: torch.Tensor,
        p3_hw: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        encoded = self.encode_child_map(child_map, batch_ids3, flat_indices3, p3_hw)
        if m2_pre_logits.ndim != 4 or m2_pre_logits.size(1) != 1:
            raise ValueError(
                f"m2_pre_logits must be [B,1,H,W], got {tuple(m2_pre_logits.shape)}"
            )
        if m2_pre_logits.size(0) != child_map.size(0):
            raise ValueError("m2_pre_logits batch must match child_map")
        if m2_pre_logits.shape[-2:] != child_map.shape[-2:]:
            m2_pre_logits = F.interpolate(
                m2_pre_logits,
                size=child_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        geometry_map = geometry_map_from_logits(m2_pre_logits)
        geometry_base = gather_tokens(
            geometry_map,
            batch_ids3,
            encoded["flat_indices2_from_p3"],
        ).to(dtype=encoded["q_child"].dtype)
        delta = self.geo_residual(torch.cat((encoded["q_child"], geometry_base), dim=-1))

        sdf = torch.tanh(geometry_base[:, 0:1] + delta[:, 0:1])
        normal = F.normalize(
            geometry_base[:, 1:3] + delta[:, 1:3],
            dim=-1,
            eps=1.0e-6,
        )
        offset = torch.tanh(geometry_base[:, 3:5] + delta[:, 3:5])
        reliability_logit = torch.logit(geometry_base[:, 5:6].clamp(1.0e-4, 1.0 - 1.0e-4))
        reliability = torch.sigmoid(reliability_logit + delta[:, 5:6])
        encoded["G2_query"] = torch.cat((sdf, normal, offset, reliability), dim=-1)
        encoded["geometry_base"] = geometry_base
        return encoded


ChildQueryBuilder = DinoChildQueryBuilder


__all__ = ["ChildQueryBuilder", "DinoChildQueryBuilder"]

