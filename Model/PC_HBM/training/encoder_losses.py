"""Losses used only by the encoder-side PC-HBM profile."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F


def _target_like(target: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError(f"Target must be [B,1,H,W], got {tuple(target.shape)}.")
    if target.shape[0] != reference.shape[0]:
        raise ValueError("Target and prediction batch sizes differ.")
    resized = F.interpolate(target.float(), size=reference.shape[-2:], mode="nearest")
    return resized.to(device=reference.device, dtype=reference.dtype).clamp(0.0, 1.0)


def encoder_bootstrap_loss(
    *,
    coarse_logits: torch.Tensor,
    boundary_logits: torch.Tensor,
    mask_target: torch.Tensor,
    boundary_target: torch.Tensor,
    coarse_weight: float = 0.30,
    boundary_weight: float = 0.10,
) -> Mapping[str, torch.Tensor]:
    """Supervise raw logits; sigmoid probabilities are never passed here."""

    if coarse_logits.ndim != 4 or coarse_logits.shape[1] != 1:
        raise ValueError("coarse_logits must be [B,1,H,W].")
    if boundary_logits.shape != coarse_logits.shape:
        raise ValueError("boundary_logits must match coarse_logits shape.")
    coarse_target = _target_like(mask_target, coarse_logits)
    resized_boundary = _target_like(boundary_target, boundary_logits)
    coarse = F.binary_cross_entropy_with_logits(coarse_logits.float(), coarse_target.float())
    boundary = F.binary_cross_entropy_with_logits(
        boundary_logits.float(), resized_boundary.float()
    )
    total = float(coarse_weight) * coarse + float(boundary_weight) * boundary
    return {"total": total, "coarse": coarse, "boundary": boundary}


__all__ = ["encoder_bootstrap_loss"]
