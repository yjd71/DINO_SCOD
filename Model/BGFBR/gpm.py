"""DINO-grid adaptation of FEDER's global perception module."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvBNReLU, make_norm
from .contracts import DEFAULT_GPM_DILATIONS


class PositionAttentionModule(nn.Module):
    """Spatial self-attention with a zero-initialized residual scale."""

    def __init__(self, channels: int = 128, query_channels: int = 16) -> None:
        super().__init__()
        if channels <= 0 or query_channels <= 0:
            raise ValueError("channels and query_channels must be positive")
        self.query = nn.Conv2d(channels, query_channels, kernel_size=1)
        self.key = nn.Conv2d(channels, query_channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.scale = query_channels**-0.5

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = feature.shape
        query = self.query(feature).flatten(2).transpose(1, 2)
        key = self.key(feature).flatten(2)
        attention = torch.softmax(torch.bmm(query, key) * self.scale, dim=-1)
        value = self.value(feature).flatten(2)
        attended = torch.bmm(value, attention.transpose(1, 2)).view(batch, channels, height, width)
        return feature + self.gamma.to(dtype=feature.dtype) * attended


class DinoGlobalPerceptionModule(nn.Module):
    """Five-branch global/multi-dilation context encoder for 28x28 DINO maps."""

    def __init__(
        self,
        channels: int = 128,
        dilations: tuple[int, int, int] = DEFAULT_GPM_DILATIONS,
        *,
        norm: str = "bn",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if len(dilations) != 3 or any(int(d) <= 0 for d in dilations):
            raise ValueError("dilations must contain three positive values")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")
        self.channels = int(channels)
        self.dilations = tuple(int(d) for d in dilations)

        # Avoid normalization on a pooled 1x1 tensor so batch-size-one training
        # remains valid.
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.point_branch = ConvBNReLU(channels, channels, kernel_size=1, padding=0, norm=norm)
        self.dilated_branches = nn.ModuleList(
            ConvBNReLU(channels, channels, kernel_size=3, dilation=d, norm=norm)
            for d in self.dilations
        )
        self.fusion = ConvBNReLU(channels * 5, channels, kernel_size=3, norm=norm)
        self.context_attention = PositionAttentionModule(channels, query_channels=max(channels // 8, 1))

        hidden = max(channels // 2, 1)
        self.cam_head = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=3, padding=1, bias=False),
            make_norm(norm, hidden),
            nn.PReLU(hidden),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if feature.ndim != 4 or feature.shape[1] != self.channels:
            raise ValueError(
                f"GPM input must have shape [B,{self.channels},H,W], got {tuple(feature.shape)}"
            )
        spatial_size = feature.shape[-2:]
        global_context = F.interpolate(
            self.global_branch(feature),
            size=spatial_size,
            mode="bilinear",
            align_corners=False,
        )
        branches = [global_context, self.point_branch(feature)]
        branches.extend(branch(feature) for branch in self.dilated_branches)
        cam_feat = self.context_attention(self.fusion(torch.cat(branches, dim=1)))
        cam_logit = self.cam_head(cam_feat)
        return cam_feat, cam_logit


GlobalPerceptionModule = DinoGlobalPerceptionModule


__all__ = ["PositionAttentionModule", "DinoGlobalPerceptionModule", "GlobalPerceptionModule"]
