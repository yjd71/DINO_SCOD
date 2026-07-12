"""Structured correction gate with validity-aware candidate statistics."""

from __future__ import annotations

import torch
from torch import nn


def _as_column(value: torch.Tensor, name: str, count: int) -> torch.Tensor:
    if value.ndim == 1:
        value = value[:, None]
    if value.shape != (count, 1):
        raise ValueError(f"{name} must be [M] or [M,1]")
    return value


def _masked_statistics(
    scores: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    probability = torch.sigmoid(scores) * valid.to(dtype=scores.dtype)
    count = valid.sum(dim=1).clamp_min(1).to(dtype=scores.dtype)
    mean = probability.sum(dim=1) / count
    centered = (probability - mean[:, None]) * valid.to(dtype=scores.dtype)
    std = torch.sqrt(centered.square().sum(dim=1) / count)
    maximum = probability.masked_fill(~valid, -1.0).max(dim=1).values
    maximum = torch.where(valid.any(dim=1), maximum, torch.zeros_like(maximum))
    return maximum, mean, std


class StructuredGateMLP(nn.Module):
    """Combine confidence/contradiction/entropy into ``gate_pc_token``."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.residual = nn.Sequential(
            nn.Linear(12, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), 1),
        )
        self.gamma_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        confidence: torch.Tensor,
        c23: torch.Tensor,
        u_token: torch.Tensor,
        parent_entropy: torch.Tensor,
        child_entropy: torch.Tensor,
        child_scores: torch.Tensor,
        geo_scores: torch.Tensor,
        top_parent_valid: torch.Tensor | None = None,
        query_valid: torch.Tensor | None = None,
        feature_group_dropout: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if child_scores.ndim != 2 or geo_scores.shape != child_scores.shape:
            raise ValueError("child_scores and geo_scores must match [M,K]")
        count = child_scores.size(0)
        confidence = _as_column(confidence, "confidence", count)
        c23 = _as_column(c23, "c23", count)
        u_token = _as_column(u_token, "u_token", count)
        if parent_entropy.ndim == 2 and parent_entropy.size(1) == 1:
            parent_entropy = parent_entropy[:, 0]
        if child_entropy.ndim == 2 and child_entropy.size(1) == 1:
            child_entropy = child_entropy[:, 0]
        if parent_entropy.shape != (count,) or child_entropy.shape != (count,):
            raise ValueError("parent_entropy and child_entropy must be [M]")
        if top_parent_valid is None:
            top_parent_valid = torch.ones_like(child_scores, dtype=torch.bool)
        else:
            top_parent_valid = top_parent_valid.to(
                device=child_scores.device, dtype=torch.bool
            )
            if top_parent_valid.shape != child_scores.shape:
                raise ValueError("top_parent_valid must be [M,K]")
        if query_valid is None:
            query_valid = top_parent_valid.any(dim=1)
        else:
            query_valid = query_valid.to(
                device=child_scores.device, dtype=torch.bool
            )
            if query_valid.shape != (count,):
                raise ValueError("query_valid must be [M]")
        query_valid = query_valid & top_parent_valid.any(dim=1)
        valid = top_parent_valid & query_valid[:, None]

        child_max, child_mean, child_std = _masked_statistics(
            child_scores, valid
        )
        geo_max, geo_mean, _ = _masked_statistics(geo_scores, valid)
        confidence_flat = confidence[:, 0]
        contradiction = c23[:, 0]
        uncertainty = u_token[:, 0]
        features = torch.stack(
            [
                confidence_flat,
                contradiction,
                uncertainty,
                parent_entropy,
                child_entropy,
                child_max,
                child_mean,
                child_std,
                geo_max,
                geo_mean,
                (1.0 - contradiction).clamp(0.0, 1.0),
                confidence_flat * (1.0 - uncertainty),
            ],
            dim=1,
        )
        if feature_group_dropout is not None:
            if feature_group_dropout.shape not in {(12,), features.shape}:
                raise ValueError(
                    "feature_group_dropout must be [12] or [M,12]"
                )
            features = features * feature_group_dropout.to(
                device=features.device, dtype=features.dtype
            )
        base = (
            confidence_flat
            + 0.5 * child_max
            + 0.5 * geo_max
            - contradiction
            - 0.5 * parent_entropy
            - 0.5 * child_entropy
        )
        residual = self.residual(features)[:, 0]
        gate = torch.sigmoid(base + torch.tanh(self.gamma_gate) * residual)
        return gate[:, None] * query_valid[:, None].to(dtype=gate.dtype)

