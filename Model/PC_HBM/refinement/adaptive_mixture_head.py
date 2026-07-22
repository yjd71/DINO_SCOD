"""Final 98 x 98 keep/residual/deformation/suppression mixture."""

from __future__ import annotations

import math
from typing import Dict, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from .boundary_deformation import deform_logits
from ..common.utils import gradient_strength


class AdaptiveMixtureHead(nn.Module):
    """Mix four logit branches without creating higher-resolution graphs."""

    def __init__(
        self,
        r_max: float = 2.0,
        max_offset: float = 1.5,
        mask_corr_epsilon: float = 0.10,
        init_bias: list[float] | tuple[float, ...] = (1.0, -0.5, -0.5, -0.5),
        use_branch_quality: bool = True,
        use_branch_dropout: bool = True,
        context_ch: int = 14,
    ) -> None:
        super().__init__()
        if len(init_bias) != 4:
            raise ValueError("init_bias must contain four branch biases")
        self.r_max = float(r_max)
        self.max_offset = float(max_offset)
        self.mask_corr_epsilon = float(mask_corr_epsilon)
        self.use_branch_dropout = bool(use_branch_dropout)
        self.context_ch = int(context_ch)
        if self.context_ch != 14:
            raise ValueError("mixture context_ch is fixed to 14 for the original Decoder")
        self.mix_head = nn.Sequential(
            nn.Conv2d(self.context_ch, 32, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 4, 1),
        )
        nn.init.zeros_(self.mix_head[-1].weight)
        with torch.no_grad():
            self.mix_head[-1].bias.copy_(
                torch.as_tensor(init_bias, dtype=self.mix_head[-1].bias.dtype)
            )
        if use_branch_quality:
            self.quality_head: nn.Module | None = nn.Sequential(
                nn.Conv2d(self.context_ch, 32, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(32, 4, 1),
            )
            nn.init.zeros_(self.quality_head[-1].weight)
            nn.init.zeros_(self.quality_head[-1].bias)
        else:
            self.quality_head = None

    def forward(
        self,
        z_main: torch.Tensor,
        p1_aux: Mapping[str, torch.Tensor],
        pc_maps: Mapping[str, torch.Tensor],
        epoch: int | None = None,
        temperature: float = 1.0,
        eps_floor: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        del epoch
        if z_main.ndim != 4 or z_main.shape[1:] != (1, 98, 98):
            raise ValueError(
                f"adaptive mixture must run at [B,1,98,98], got {tuple(z_main.shape)}"
            )
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= eps_floor < 0.25:
            raise ValueError("eps_floor must be in [0, 0.25)")
        size = (98, 98)
        probability = torch.sigmoid(z_main)
        boundary = F.interpolate(
            p1_aux["B1"], size=size, mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
        gate = F.interpolate(
            p1_aux["G1_map"], size=size, mode="bilinear", align_corners=False
        ).clamp(0.0, 1.0)
        valid = F.interpolate(
            p1_aux["valid1_map"], size=size, mode="nearest"
        ).clamp(0.0, 1.0)

        # P1 exports raw values. Each activation/amplitude limit is applied
        # exactly once here, and suppression is zero-centred at raw == 0.
        residual_raw = F.interpolate(
            p1_aux["R1_map"], size=size, mode="bilinear", align_corners=False
        )
        offset_raw = F.interpolate(
            p1_aux["O1_map"], size=size, mode="bilinear", align_corners=False
        )
        suppress_raw = F.interpolate(
            p1_aux["R_sup_map"], size=size, mode="bilinear", align_corners=False
        )
        residual = torch.tanh(residual_raw) * self.r_max
        offset = torch.tanh(offset_raw) * self.max_offset
        softplus_zero = F.softplus(torch.zeros_like(suppress_raw))
        suppression = F.softplus(suppress_raw) - softplus_zero
        correction_mask = valid * gate * (
            self.mask_corr_epsilon
            + (1.0 - self.mask_corr_epsilon) * boundary
        )

        z_keep = z_main
        z_residual = z_main + correction_mask * residual
        sampled = deform_logits(z_main, offset, correction_mask)
        identity_sample = deform_logits(
            z_main, torch.zeros_like(offset), torch.zeros_like(correction_mask)
        )
        # Correct the tiny grid_sample identity error without blocking offset
        # gradients. At zero initialization this is bitwise z_main.
        z_deformed = sampled + (z_main - identity_sample).detach()
        z_suppressed = z_main - correction_mask * suppression

        uncertainty = 4.0 * probability * (1.0 - probability)
        gradient = gradient_strength(probability)
        c23 = F.interpolate(
            pc_maps["C23_map"], size=size, mode="bilinear", align_corners=False
        )
        mask_pc = F.interpolate(
            pc_maps["M_pc_map"], size=size, mode="bilinear", align_corners=False
        )
        offset_magnitude = torch.linalg.vector_norm(
            offset, dim=1, keepdim=True
        ) / max(self.max_offset, 1.0e-6)
        context = torch.cat(
            [
                probability,
                boundary,
                gate,
                correction_mask,
                uncertainty,
                gradient,
                c23,
                mask_pc,
                offset,
                residual,
                suppression,
                valid,
                offset_magnitude,
            ],
            dim=1,
        )
        if context.size(1) != self.context_ch:
            raise RuntimeError(
                f"mixture context expected {self.context_ch} channels, got {context.size(1)}"
            )
        mix_logits = self.mix_head(context)
        if self.training and self.use_branch_dropout:
            drop = torch.rand(
                mix_logits.size(0),
                4,
                1,
                1,
                device=mix_logits.device,
                dtype=mix_logits.dtype,
            )
            mix_logits = mix_logits.masked_fill(drop < 0.02, -1.0e4)
        mixture = torch.softmax(mix_logits / float(temperature), dim=1)
        if eps_floor > 0:
            mixture = (1.0 - 4.0 * eps_floor) * mixture + eps_floor
            mixture = mixture / mixture.sum(dim=1, keepdim=True).clamp_min(
                1.0e-6
            )
        branches = torch.cat(
            [z_keep, z_residual, z_deformed, z_suppressed], dim=1
        )
        # Difference form guarantees exact identity when all branches equal
        # z_main, independently of mixture probabilities and summation error.
        z_final = z_main + (
            mixture * (branches - z_main)
        ).sum(dim=1, keepdim=True)
        quality = (
            self.quality_head(context)
            if self.quality_head is not None
            else torch.zeros_like(mixture)
        )
        return {
            "z_keep": z_keep,
            "z_res": z_residual,
            "z_def": z_deformed,
            "z_sup": z_suppressed,
            "z_warp": z_deformed,
            "pi": mixture,
            "mix_logits": mix_logits,
            "pred_gain": quality,
            "branch_quality": quality,
            "B_pix": boundary,
            "G_pix": gate,
            "Mask_corr": correction_mask,
            "R_pix": residual,
            "O_pix": offset,
            "R_sup": suppression,
            "valid_pix": valid,
            "z_final": z_final,
            "p_final": torch.sigmoid(z_final),
        }
