"""Adaptive-mixture oracle targets from the validated PC-HBM formulation."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F


BRANCH_NAMES = ("z_keep", "z_res", "z_def", "z_sup")


def branch_errors(branch_logits: Mapping[str, torch.Tensor], gt: torch.Tensor) -> torch.Tensor:
    """Return BCE plus absolute probability error as ``[B,4,H,W]``."""

    missing = [name for name in BRANCH_NAMES if name not in branch_logits]
    if missing:
        raise KeyError(f"Missing mixture branches: {missing}")
    reference = branch_logits["z_keep"]
    if gt.ndim == 3:
        gt = gt.unsqueeze(1)
    target = F.interpolate(gt.float(), size=reference.shape[-2:], mode="nearest")
    errors = []
    for name in BRANCH_NAMES:
        logits = branch_logits[name]
        if logits.shape != reference.shape:
            raise ValueError(
                f"All mixture branches must share shape {tuple(reference.shape)}, "
                f"but {name} is {tuple(logits.shape)}"
            )
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        errors.append(bce + (torch.sigmoid(logits) - target).abs())
    return torch.cat(errors, dim=1)


def oracle_distribution(
    branch_logits: Mapping[str, torch.Tensor],
    gt: torch.Tensor,
    *,
    tau: float = 0.5,
    min_improvement: float = 0.03,
) -> dict[str, torch.Tensor]:
    """Build the soft best-branch distribution and useful-refinement mask."""

    errors = branch_errors(branch_logits, gt)
    target_mix = torch.softmax(-errors / max(float(tau), 1.0e-6), dim=1)
    keep_error = errors[:, 0:1]
    improvement = keep_error - errors.min(dim=1, keepdim=True).values
    boundary_weight = branch_logits.get("B_pix")
    if boundary_weight is None:
        boundary_weight = torch.ones_like(keep_error)
    elif boundary_weight.shape[-2:] != errors.shape[-2:]:
        boundary_weight = F.interpolate(
            boundary_weight,
            size=errors.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    oracle_mask = (improvement > float(min_improvement)).to(errors.dtype) * boundary_weight.detach()
    return {
        "pixel_error": errors,
        "target_mix": target_mix,
        "oracle_mask": oracle_mask,
        "improvement": improvement,
    }


__all__ = ["BRANCH_NAMES", "branch_errors", "oracle_distribution"]
