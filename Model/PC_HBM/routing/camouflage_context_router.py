"""Parameter-free global/environment routing for PC-HBM-Lite."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class CamouflageContextRouter(nn.Module):
    """Build two normalized contexts and use them only for image routing."""

    def __init__(
        self,
        dim: int = 128,
        top_img_k: int = 4,
        global_weight: float = 0.5,
        environment_weight: float = 0.5,
        min_environment_mass: float = 1.0e-3,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.top_img_k = int(top_img_k)
        self.global_weight = float(global_weight)
        self.environment_weight = float(environment_weight)
        self.min_environment_mass = float(min_environment_mass)
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.top_img_k <= 0:
            raise ValueError("top_img_k must be positive")
        if (
            not math.isfinite(self.global_weight)
            or not math.isfinite(self.environment_weight)
            or self.global_weight < 0.0
            or self.environment_weight < 0.0
            or max(self.global_weight, self.environment_weight) <= 0.0
        ):
            raise ValueError(
                "route weights must be finite, non-negative, and not both zero"
            )
        if (
            not math.isfinite(self.min_environment_mass)
            or self.min_environment_mass < 0
        ):
            raise ValueError("min_environment_mass must be non-negative")

    def encode_route_tokens(
        self,
        x3: torch.Tensor,
        prob3: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return normalized global and environment descriptors in FP32."""

        if x3.ndim != 4 or x3.size(1) != self.dim:
            raise ValueError(f"x3 must be [B,{self.dim},H,W], got {tuple(x3.shape)}")
        if (
            prob3.ndim != 4
            or prob3.size(1) != 1
            or prob3.size(0) != x3.size(0)
            or prob3.shape[-2:] != x3.shape[-2:]
        ):
            raise ValueError(
                "prob3 must be [B,1,H,W] and share x3 batch/spatial dimensions, "
                f"got {tuple(prob3.shape)} for x3 {tuple(x3.shape)}"
            )

        feature = torch.nan_to_num(x3.float())
        probability = torch.nan_to_num(prob3.float(), nan=0.5).clamp(0.0, 1.0)
        route_global = F.normalize(feature.mean(dim=(-2, -1)), dim=-1, eps=1.0e-6)

        uncertainty = 4.0 * probability * (1.0 - probability)
        dx = F.pad(probability[..., 1:] - probability[..., :-1], (0, 1, 0, 0))
        dy = F.pad(probability[..., 1:, :] - probability[..., :-1, :], (0, 0, 0, 1))
        gradient = torch.sqrt(dx.square() + dy.square()).clamp(0.0, 1.0)
        environment_mask = (1.0 - probability) * (1.0 - uncertainty) * (1.0 - gradient)
        fallback_mask = 1.0 - probability

        primary_mass = environment_mask.sum(dim=(-2, -1), keepdim=True)
        fallback_mass = fallback_mask.sum(dim=(-2, -1), keepdim=True)
        use_primary = primary_mass >= self.min_environment_mass
        use_fallback = (~use_primary) & (fallback_mass >= self.min_environment_mass)
        full_mask = torch.ones_like(environment_mask)
        selected_mask = torch.where(
            use_primary,
            environment_mask,
            torch.where(use_fallback, fallback_mask, full_mask),
        )
        denominator = selected_mask.sum(dim=(-2, -1)).clamp_min(1.0e-6)
        pooled = (feature * selected_mask).sum(dim=(-2, -1)) / denominator
        route_environment = F.normalize(
            torch.nan_to_num(pooled),
            dim=-1,
            eps=1.0e-6,
        )
        return {
            "route_global": route_global,
            "route_environment": route_environment,
        }

    def forward(
        self,
        x3: torch.Tensor,
        prob3: torch.Tensor,
        memory: Any,
        top_img_k: int | None = None,
        *,
        query_image_ids: Sequence[object] | None = None,
        exclude_self_match: bool = True,
    ) -> dict[str, Any]:
        contexts = self.encode_route_tokens(x3, prob3)
        routed = memory.route_query(
            q_global=contexts["route_global"],
            q_environment=contexts["route_environment"],
            top_img_k=self.top_img_k if top_img_k is None else int(top_img_k),
            query_image_ids=query_image_ids,
            exclude_self_match=exclude_self_match,
            global_weight=self.global_weight,
            environment_weight=self.environment_weight,
        )
        routed.update(contexts)
        return routed


__all__ = ["CamouflageContextRouter"]
