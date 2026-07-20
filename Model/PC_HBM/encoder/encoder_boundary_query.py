"""Boundary evidence and sparse token selection for encoder-side retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from .feature_projector import ProjectedDinoFeatures


def _spatial_gradient(value: torch.Tensor) -> torch.Tensor:
    if value.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got {tuple(value.shape)}.")
    dx = F.pad(value[..., 1:] - value[..., :-1], (0, 1, 0, 0))
    dy = F.pad(value[..., 1:, :] - value[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + 1e-12)


def _normalize_per_sample(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    scale = value.flatten(1).amax(dim=1).clamp_min(eps)
    return value / scale[:, None, None, None]


@dataclass(frozen=True)
class EncoderBoundaryOutput:
    boundary_logits: torch.Tensor
    boundary_probability: torch.Tensor
    selected_indices: torch.Tensor
    selected_mask: torch.Tensor
    evidence_channels: torch.Tensor


class EncoderBoundaryQuery(nn.Module):
    def __init__(
        self,
        memory_dim: int = 128,
        token_ratio: float = 0.20,
        min_tokens: int = 32,
        max_tokens: int = 128,
    ) -> None:
        super().__init__()
        if not 0.0 < token_ratio <= 1.0:
            raise ValueError("token_ratio must be in (0, 1].")
        if min_tokens <= 0 or max_tokens < min_tokens:
            raise ValueError("Invalid boundary token limits.")
        self.memory_dim = int(memory_dim)
        self.token_ratio = float(token_ratio)
        self.min_tokens = int(min_tokens)
        self.max_tokens = int(max_tokens)
        self.head = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def _evidence(
        self,
        features: ProjectedDinoFeatures,
        coarse_probability: torch.Tensor,
    ) -> torch.Tensor:
        f1, f2, f3, _ = features.maps
        if coarse_probability.shape != (
            f1.shape[0],
            1,
            f1.shape[2],
            f1.shape[3],
        ):
            raise ValueError("coarse_probability is incompatible with DINO maps.")
        probability = coarse_probability.float().clamp(1e-6, 1.0 - 1e-6)
        uncertainty = 4.0 * probability * (1.0 - probability)
        probability_gradient = _normalize_per_sample(_spatial_gradient(probability))
        entropy = -(
            probability * probability.log()
            + (1.0 - probability) * (1.0 - probability).log()
        ) / math.log(2.0)
        f1_gradient = _normalize_per_sample(
            _spatial_gradient(f1.float()).square().mean(dim=1, keepdim=True).sqrt()
        )
        disagreement = 1.0 - F.cosine_similarity(
            f2.float(), f3.float(), dim=1, eps=1e-6
        ).unsqueeze(1)
        disagreement = (0.5 * disagreement).clamp(0.0, 1.0)
        return torch.cat(
            (
                probability,
                uncertainty,
                probability_gradient,
                entropy,
                f1_gradient,
                disagreement,
            ),
            dim=1,
        ).to(dtype=f1.dtype)

    def forward(
        self,
        features: ProjectedDinoFeatures,
        coarse_probability: torch.Tensor,
    ) -> EncoderBoundaryOutput:
        evidence = self._evidence(features, coarse_probability)
        logits = self.head(evidence)
        probability = torch.sigmoid(logits)
        token_count = probability.shape[-2] * probability.shape[-1]
        selected_count = max(
            self.min_tokens,
            min(self.max_tokens, int(math.ceil(token_count * self.token_ratio))),
        )
        selected_count = min(selected_count, token_count)
        indices = probability.flatten(1).topk(selected_count, dim=1).indices
        mask = torch.ones_like(indices, dtype=torch.bool)
        return EncoderBoundaryOutput(
            boundary_logits=logits,
            boundary_probability=probability,
            selected_indices=indices,
            selected_mask=mask,
            evidence_channels=evidence,
        )


__all__ = ["EncoderBoundaryOutput", "EncoderBoundaryQuery"]
