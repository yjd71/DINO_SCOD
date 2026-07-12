"""Differentiable logit deformation used by the final mixture."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..common.utils import make_normalized_grid


def deform_logits(
    z_main: torch.Tensor,
    offset_pix: torch.Tensor,
    mask_corr: torch.Tensor,
) -> torch.Tensor:
    """Warp ``z_main`` with pixel-unit offsets and ``align_corners=False``."""

    if z_main.ndim != 4 or z_main.size(1) != 1:
        raise ValueError(f"z_main must be [B,1,H,W], got {tuple(z_main.shape)}")
    if offset_pix.shape != (z_main.size(0), 2, *z_main.shape[-2:]):
        raise ValueError("offset_pix must be [B,2,H,W] and match z_main")
    if mask_corr.shape != z_main.shape:
        raise ValueError("mask_corr must match z_main")
    batch_size, _, height, width = z_main.shape
    grid = make_normalized_grid(
        height, width, z_main.device, z_main.dtype
    ).expand(batch_size, height, width, 2)
    x_offset = offset_pix[:, 0:1] * mask_corr * (2.0 / width)
    y_offset = offset_pix[:, 1:2] * mask_corr * (2.0 / height)
    delta = torch.cat((x_offset, y_offset), dim=1).permute(0, 2, 3, 1)
    return F.grid_sample(
        z_main,
        grid + delta,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )

