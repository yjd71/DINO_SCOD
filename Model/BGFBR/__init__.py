"""Boundary-guided foreground/background refinement primitives."""

from .adapter import F4Adapter
from .bridges import P3P2CorrectionBridge
from .common import ConvBNReLU, make_norm
from .contracts import (
    BGFBR_CONTRACT_VERSION,
    DEFAULT_GPM_DILATIONS,
    FOREGROUND_BACKGROUND_CONTRACT_VERSION,
    GBE_NORMALIZATION_VERSION,
    BGFBRStageOutput,
    StageOutput,
)
from .gpm import DinoGlobalPerceptionModule, GlobalPerceptionModule, PositionAttentionModule
from .image_rgb import GradientBoundaryEnhancement, ImageNetRGBAdapter
from .ode import ODEBlock, OrthogonalDualEnhancement
from .rcab import ChannelAttention, RCAB, ResidualChannelAttentionBlock
from .stage import BGFBRStage, BgPriorEmbed, EdgeEmbed, FinalFusion

__all__ = [
    "BGFBR_CONTRACT_VERSION",
    "FOREGROUND_BACKGROUND_CONTRACT_VERSION",
    "GBE_NORMALIZATION_VERSION",
    "DEFAULT_GPM_DILATIONS",
    "StageOutput",
    "BGFBRStageOutput",
    "ConvBNReLU",
    "make_norm",
    "ImageNetRGBAdapter",
    "GradientBoundaryEnhancement",
    "F4Adapter",
    "PositionAttentionModule",
    "DinoGlobalPerceptionModule",
    "GlobalPerceptionModule",
    "ODEBlock",
    "OrthogonalDualEnhancement",
    "ChannelAttention",
    "RCAB",
    "ResidualChannelAttentionBlock",
    "EdgeEmbed",
    "BgPriorEmbed",
    "BGFBRStage",
    "FinalFusion",
    "P3P2CorrectionBridge",
]
