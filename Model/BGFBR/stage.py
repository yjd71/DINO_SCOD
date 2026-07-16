"""Foreground/background refinement stage and final decoder fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ConvBNReLU
from .contracts import StageOutput
from .ode import ODEBlock
from .rcab import RCAB


class EdgeEmbed(nn.Module):
    """Explicit learnable one-channel boundary embedding."""

    def __init__(self, channels: int = 128, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        # No bias: a disabled/zero GBE context must contribute exact zero.
        self.projection = nn.Conv2d(1, channels, kernel_size, padding=padding, bias=False)

    def forward(self, edge: torch.Tensor) -> torch.Tensor:
        if edge.ndim != 4 or edge.shape[1] != 1:
            raise ValueError(f"edge must have shape [B,1,H,W], got {tuple(edge.shape)}")
        return self.projection(edge)


class BgPriorEmbed(nn.Module):
    """Explicit learnable embedding of the complementary CAM prior."""

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.projection = nn.Conv2d(1, channels, kernel_size=1, bias=False)

    def forward(self, background_prior: torch.Tensor) -> torch.Tensor:
        if background_prior.ndim != 4 or background_prior.shape[1] != 1:
            raise ValueError(
                "background_prior must have shape [B,1,H,W], "
                f"got {tuple(background_prior.shape)}"
            )
        return self.projection(background_prior)


class BGFBRStage(nn.Module):
    """One independently parameterized foreground/background refinement stage."""

    def __init__(
        self,
        channels: int = 128,
        *,
        reduction: int = 16,
        norm: str = "bn",
        use_ode: bool = True,
        use_rcab: bool = True,
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.use_ode = bool(use_ode)
        self.use_rcab = bool(use_rcab)

        self.ode_input = ConvBNReLU(channels * 2, channels, kernel_size=1, padding=0, norm=norm)
        self.ode = ODEBlock(channels)

        self.fg_input = ConvBNReLU(channels * 2, channels, kernel_size=1, padding=0, norm=norm)
        self.fg_rcab = RCAB(channels, reduction)
        self.fg_refine = ConvBNReLU(channels, channels, kernel_size=3, norm=norm)

        self.bg_input = ConvBNReLU(channels * 2, channels, kernel_size=1, padding=0, norm=norm)
        self.bg_rcab = RCAB(channels, reduction)
        self.bg_refine = ConvBNReLU(channels, channels, kernel_size=3, norm=norm)

        self.edge_embed = EdgeEmbed(channels)
        self.bg_prior_embed = BgPriorEmbed(channels)
        self.fg_head = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.bg_head = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

    def _validate_inputs(
        self,
        feature: torch.Tensor,
        cam_feat: torch.Tensor,
        cam_logit: torch.Tensor,
        edge_28: torch.Tensor,
    ) -> None:
        expected_feature = (feature.shape[0], self.channels, *feature.shape[-2:])
        if feature.ndim != 4 or feature.shape[1] != self.channels:
            raise ValueError(
                f"feature must have shape [B,{self.channels},H,W], got {tuple(feature.shape)}"
            )
        if tuple(cam_feat.shape) != expected_feature:
            raise ValueError(
                f"cam_feat must match feature shape {expected_feature}, got {tuple(cam_feat.shape)}"
            )
        expected_single = (feature.shape[0], 1, *feature.shape[-2:])
        if tuple(cam_logit.shape) != expected_single:
            raise ValueError(
                f"cam_logit must have shape {expected_single}, got {tuple(cam_logit.shape)}"
            )
        if tuple(edge_28.shape) != expected_single:
            raise ValueError(f"edge_28 must have shape {expected_single}, got {tuple(edge_28.shape)}")

    def forward(
        self,
        feature: torch.Tensor,
        cam_feat: torch.Tensor,
        cam_logit: torch.Tensor,
        edge_28: torch.Tensor,
    ) -> StageOutput:
        self._validate_inputs(feature, cam_feat, cam_logit, edge_28)
        cam_prob = torch.sigmoid(cam_logit)
        ode_input = self.ode_input(torch.cat((feature, cam_feat), dim=1))
        ode_feature = self.ode(ode_input) if self.use_ode else ode_input

        coarse_fg = cam_prob * feature
        fg_body = self.fg_input(torch.cat((coarse_fg, ode_feature), dim=1))
        if self.use_rcab:
            fg_body = self.fg_rcab(fg_body)
        fg_feature = self.fg_refine(fg_body + cam_feat) + self.edge_embed(edge_28)
        fg_logit = self.fg_head(fg_feature)

        bg_prior = 1.0 - cam_prob
        coarse_bg = bg_prior * feature
        bg_body = self.bg_input(torch.cat((coarse_bg, ode_feature), dim=1))
        if self.use_rcab:
            bg_body = self.bg_rcab(bg_body)
        bg_feature = (
            self.bg_refine(bg_body)
            + self.bg_prior_embed(bg_prior)
            + self.edge_embed(edge_28)
        )
        bg_logit = self.bg_head(bg_feature)
        dual_uncertainty = 1.0 - (torch.sigmoid(fg_logit) - torch.sigmoid(bg_logit)).abs()

        return StageOutput(
            fg_feature=fg_feature,
            bg_feature=bg_feature,
            fg_logit=fg_logit,
            bg_logit=bg_logit,
            dual_uncertainty=dual_uncertainty,
        )


class FinalFusion(nn.Module):
    """Fuse Stage1, refined P2, CAM and RGB edge context into final logits."""

    def __init__(
        self,
        channels: int = 128,
        output_size: tuple[int, int] = (98, 98),
        *,
        norm: str = "bn",
    ) -> None:
        super().__init__()
        self.channels = int(channels)
        self.output_size = tuple(int(v) for v in output_size)
        self.edge_embed_28 = EdgeEmbed(channels)
        self.edge_embed_98 = EdgeEmbed(channels)
        self.fusion = ConvBNReLU(channels * 4, channels, kernel_size=3, norm=norm)
        self.final_head = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

    def forward(
        self,
        fg1_feature: torch.Tensor,
        p2_refined: torch.Tensor,
        cam_feat: torch.Tensor,
        edge_28: torch.Tensor,
        edge_98: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if fg1_feature.ndim != 4 or fg1_feature.shape[1] != self.channels:
            raise ValueError(
                f"fg1_feature must have shape [B,{self.channels},H,W], "
                f"got {tuple(fg1_feature.shape)}"
            )
        if p2_refined.shape != fg1_feature.shape or cam_feat.shape != fg1_feature.shape:
            raise ValueError("p2_refined and cam_feat must exactly match fg1_feature")
        expected_edge_28 = (fg1_feature.shape[0], 1, *fg1_feature.shape[-2:])
        if tuple(edge_28.shape) != expected_edge_28:
            raise ValueError(f"edge_28 must have shape {expected_edge_28}, got {tuple(edge_28.shape)}")
        expected_edge_98 = (fg1_feature.shape[0], 1, *self.output_size)
        if tuple(edge_98.shape) != expected_edge_98:
            raise ValueError(f"edge_98 must have shape {expected_edge_98}, got {tuple(edge_98.shape)}")

        embedded_edge_28 = self.edge_embed_28(edge_28)
        p1_28 = self.fusion(torch.cat((fg1_feature, p2_refined, cam_feat, embedded_edge_28), dim=1))
        p1_98 = F.interpolate(p1_28, size=self.output_size, mode="bilinear", align_corners=False)
        p1_98 = p1_98 + self.edge_embed_98(edge_98)
        z_main = self.final_head(p1_98)
        return p1_28, p1_98, z_main


__all__ = ["EdgeEmbed", "BgPriorEmbed", "BGFBRStage", "FinalFusion"]
