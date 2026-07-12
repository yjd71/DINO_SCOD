"""Five-descriptor camouflage context routing for labelled image memory."""

from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.utils import EPS, finite_or_zero, gradient_strength, normalize
from .route_attention_pool import RouteAttentionPool


class CamouflageContextRouter(nn.Module):
    def __init__(self, x3_ch: int, dim: int = 128, top_img_k: int = 8) -> None:
        super().__init__()
        self.dim = int(dim)
        self.top_img_k = int(top_img_k)
        self.proj_x3 = nn.Conv2d(int(x3_ch), self.dim, kernel_size=1, bias=False)
        self.pool = RouteAttentionPool(self.dim)

    def encode_route_tokens(
        self,
        x3: torch.Tensor,
        prob3: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if x3.ndim != 4:
            raise ValueError(f"x3 must be [B,C,H,W], got {tuple(x3.shape)}")
        projected = self.proj_x3(x3)
        if prob3 is None:
            probability = torch.sigmoid(projected[:, :1])
        else:
            if prob3.ndim != 4 or prob3.size(1) != 1 or prob3.size(0) != x3.size(0):
                raise ValueError(f"prob3 must be [B,1,H,W], got {tuple(prob3.shape)}")
            probability = F.interpolate(
                prob3,
                size=projected.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).clamp(0.0, 1.0)
        uncertainty = 4.0 * probability * (1.0 - probability)
        gradient = gradient_strength(probability)
        near_background = (probability < 0.35).to(dtype=projected.dtype)
        environment = 1.0 - gradient.clamp(0.0, 1.0)
        descriptors = {
            "x3_global": self._masked_pool(projected, torch.ones_like(probability)),
            "x3_boundary": self._masked_pool(projected, gradient + uncertainty),
            "x3_uncertain": self._masked_pool(projected, uncertainty),
            "x3_bg_near": self._masked_pool(projected, near_background * (gradient + 0.25)),
            "x3_environment": self._masked_pool(projected, environment),
        }
        order = (
            "x3_global",
            "x3_boundary",
            "x3_uncertain",
            "x3_bg_near",
            "x3_environment",
        )
        stacked = torch.stack([descriptors[name] for name in order], dim=1)
        route_embed, route_weights = self.pool(stacked)
        descriptors["route_embed"] = route_embed
        descriptors["route_weights"] = route_weights
        return descriptors

    def forward(
        self,
        x3: torch.Tensor,
        prob3: torch.Tensor,
        memory,
        top_img_k: int | None = None,
        *,
        query_image_ids: Sequence[object] | None = None,
        exclude_self_match: bool = True,
    ) -> Dict[str, object]:
        route_tokens = self.encode_route_tokens(x3, prob3)
        routed = memory.route_query(
            route_tokens["route_embed"],
            self.top_img_k if top_img_k is None else int(top_img_k),
            query_image_ids=query_image_ids,
            exclude_self_match=exclude_self_match,
        )
        routed["route_context"] = route_tokens["route_embed"]
        routed["route_tokens"] = route_tokens
        # Keep the descriptor keys available at the top level for the DINO
        # engine and for memory-entry construction diagnostics.
        routed.update(route_tokens)
        return routed

    @staticmethod
    def _masked_pool(feature: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = F.interpolate(mask, size=feature.shape[-2:], mode="bilinear", align_corners=False).clamp_min(0.0)
        denominator = mask.sum(dim=(-2, -1), keepdim=True).clamp_min(EPS)
        pooled = (feature * mask).sum(dim=(-2, -1), keepdim=True) / denominator
        return normalize(finite_or_zero(pooled.flatten(1)), dim=-1)


__all__ = ["CamouflageContextRouter"]
