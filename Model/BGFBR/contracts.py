"""Stable tensor contracts shared by the BGFBR decoder components."""

from __future__ import annotations

from dataclasses import dataclass

import torch


BGFBR_CONTRACT_VERSION = 1
FOREGROUND_BACKGROUND_CONTRACT_VERSION = 1
GBE_NORMALIZATION_VERSION = "sobel_rgb_replicate_maxnorm_v1"
DEFAULT_GPM_DILATIONS = (1, 3, 5)


@dataclass(frozen=True)
class StageOutput:
    """Native-scale foreground/background output of one BGFBR stage."""

    fg_feature: torch.Tensor
    bg_feature: torch.Tensor
    fg_logit: torch.Tensor
    bg_logit: torch.Tensor
    dual_uncertainty: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "fg_feature": self.fg_feature,
            "bg_feature": self.bg_feature,
            "fg_logit": self.fg_logit,
            "bg_logit": self.bg_logit,
            "dual_uncertainty": self.dual_uncertainty,
        }


BGFBRStageOutput = StageOutput


__all__ = [
    "BGFBR_CONTRACT_VERSION",
    "FOREGROUND_BACKGROUND_CONTRACT_VERSION",
    "GBE_NORMALIZATION_VERSION",
    "DEFAULT_GPM_DILATIONS",
    "StageOutput",
    "BGFBRStageOutput",
]
