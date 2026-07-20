"""Coarse encoder-side segmentation evidence from projected DINO levels."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .feature_projector import ProjectedDinoFeatures
from .feature_projector import DinoFeatureProjector
from .encoder_boundary_query import EncoderBoundaryOutput, EncoderBoundaryQuery
from .contracts import DinoFeatureBundle


@dataclass(frozen=True)
class EncoderGlobalOutput:
    fused_map: torch.Tensor
    coarse_logits: torch.Tensor
    coarse_probability: torch.Tensor


class EncoderGlobalFusion(nn.Module):
    def __init__(self, memory_dim: int = 128) -> None:
        super().__init__()
        self.memory_dim = int(memory_dim)
        self.fusion = nn.Sequential(
            nn.Conv2d(4 * memory_dim, memory_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, memory_dim),
            nn.GELU(),
            nn.Conv2d(memory_dim, memory_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, memory_dim),
            nn.GELU(),
        )
        self.coarse_head = nn.Conv2d(memory_dim, 1, kernel_size=1)

    def forward(self, features: ProjectedDinoFeatures) -> EncoderGlobalOutput:
        maps = features.maps
        expected = self.memory_dim
        if any(feature.shape[1] != expected for feature in maps):
            raise ValueError(f"All projected maps must have {expected} channels.")
        fused = self.fusion(torch.cat(maps, dim=1))
        logits = self.coarse_head(fused)
        return EncoderGlobalOutput(
            fused_map=fused,
            coarse_logits=logits,
            coarse_probability=torch.sigmoid(logits),
        )


@dataclass(frozen=True)
class EncoderBootstrapOutput:
    projected: ProjectedDinoFeatures
    global_output: EncoderGlobalOutput
    boundary_output: EncoderBoundaryOutput


class EncoderBootstrap(nn.Module):
    """Trainable memory-space evidence that does not require a memory bank."""

    def __init__(
        self,
        *,
        encoder_dim: int = 768,
        memory_dim: int = 128,
        token_size: int = 28,
        boundary_token_ratio: float = 0.20,
        boundary_min_tokens: int = 32,
        boundary_max_tokens: int = 128,
    ) -> None:
        super().__init__()
        self.projector = DinoFeatureProjector(encoder_dim, memory_dim, token_size)
        self.global_fusion = EncoderGlobalFusion(memory_dim)
        self.boundary_query = EncoderBoundaryQuery(
            memory_dim=memory_dim,
            token_ratio=boundary_token_ratio,
            min_tokens=boundary_min_tokens,
            max_tokens=boundary_max_tokens,
        )

    def forward(self, bundle: DinoFeatureBundle) -> EncoderBootstrapOutput:
        projected = self.projector(bundle)
        global_output = self.global_fusion(projected)
        boundary_output = self.boundary_query(
            projected, global_output.coarse_probability
        )
        return EncoderBootstrapOutput(projected, global_output, boundary_output)


__all__ = [
    "EncoderBootstrap",
    "EncoderBootstrapOutput",
    "EncoderGlobalFusion",
    "EncoderGlobalOutput",
]
