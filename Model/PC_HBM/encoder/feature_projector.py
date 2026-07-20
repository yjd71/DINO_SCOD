"""DINO token projection into the 128-dimensional memory space."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .contracts import DinoFeatureBundle, Tensor4


def _projector(encoder_dim: int, memory_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(encoder_dim, memory_dim),
        nn.GELU(),
        nn.LayerNorm(memory_dim),
    )


@dataclass(frozen=True)
class ProjectedDinoFeatures:
    patch_tokens: Tensor4
    cls_tokens: Tensor4
    token_size: int

    @property
    def maps(self) -> Tensor4:
        maps = []
        for tokens in self.patch_tokens:
            batch, count, channels = tokens.shape
            expected = self.token_size * self.token_size
            if count != expected:
                raise ValueError(
                    f"Projected token count must be {expected}, got {count}."
                )
            maps.append(
                tokens.transpose(1, 2)
                .reshape(batch, channels, self.token_size, self.token_size)
                .contiguous()
            )
        return tuple(maps)  # type: ignore[return-value]


class DinoFeatureProjector(nn.Module):
    """Project each DINO patch and CLS level with independent parameters."""

    def __init__(
        self,
        encoder_dim: int = 768,
        memory_dim: int = 128,
        token_size: int = 28,
    ) -> None:
        super().__init__()
        if encoder_dim <= 0 or memory_dim <= 0 or token_size <= 0:
            raise ValueError("Projection dimensions must be positive.")
        self.encoder_dim = int(encoder_dim)
        self.memory_dim = int(memory_dim)
        self.token_size = int(token_size)
        self.patch_projectors = nn.ModuleList(
            [_projector(encoder_dim, memory_dim) for _ in range(4)]
        )
        self.cls_projectors = nn.ModuleList(
            [_projector(encoder_dim, memory_dim) for _ in range(4)]
        )

    def forward(self, bundle: DinoFeatureBundle) -> ProjectedDinoFeatures:
        bundle.validate(
            token_count=self.token_size * self.token_size,
            encoder_dim=self.encoder_dim,
        )
        patch = tuple(
            projector(tokens)
            for projector, tokens in zip(self.patch_projectors, bundle.patch_tokens)
        )
        cls = tuple(
            projector(tokens)
            for projector, tokens in zip(self.cls_projectors, bundle.cls_tokens)
        )
        return ProjectedDinoFeatures(
            patch_tokens=patch,  # type: ignore[arg-type]
            cls_tokens=cls,  # type: ignore[arg-type]
            token_size=self.token_size,
        )


__all__ = ["DinoFeatureProjector", "ProjectedDinoFeatures"]
