"""Residual bridges between corrected PC-HBM hierarchy levels."""

from __future__ import annotations

import torch
import torch.nn as nn


class P3P2CorrectionBridge(nn.Module):
    """Inject a P3 correction into P2 while initializing as exact identity."""

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.channels = int(channels)
        self.gate = nn.Conv2d(channels * 2 + 1, channels, kernel_size=3, padding=1)
        self.delta_projection = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.mask_head = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.delta_projection.weight)
        nn.init.zeros_(self.delta_projection.bias)
        nn.init.zeros_(self.mask_head.weight)
        nn.init.zeros_(self.mask_head.bias)

    def forward(
        self,
        p2_pre: torch.Tensor,
        p3_base: torch.Tensor,
        p3_corr: torch.Tensor,
        m2_pre: torch.Tensor,
        edge_28: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if p2_pre.ndim != 4 or p2_pre.shape[1] != self.channels:
            raise ValueError(
                f"p2_pre must have shape [B,{self.channels},H,W], got {tuple(p2_pre.shape)}"
            )
        if p3_base.shape != p2_pre.shape or p3_corr.shape != p2_pre.shape:
            raise ValueError("p3_base and p3_corr must exactly match p2_pre")
        expected_single = (p2_pre.shape[0], 1, *p2_pre.shape[-2:])
        if tuple(m2_pre.shape) != expected_single or tuple(edge_28.shape) != expected_single:
            raise ValueError(f"m2_pre and edge_28 must have shape {expected_single}")

        delta3 = p3_corr - p3_base
        if valid_mask is None:
            valid_mask = (delta3.detach().abs().amax(dim=1, keepdim=True) > 0).to(
                dtype=delta3.dtype
            )
        elif tuple(valid_mask.shape) != expected_single:
            raise ValueError(f"valid_mask must have shape {expected_single}")
        else:
            valid_mask = (valid_mask > 0).to(device=delta3.device, dtype=delta3.dtype)
        gate32 = torch.sigmoid(self.gate(torch.cat((p2_pre, delta3, edge_28), dim=1)))
        delta2 = self.delta_projection(delta3) * gate32 * valid_mask
        p2_pc = p2_pre + delta2
        m2_pc = m2_pre + self.mask_head(delta2) * valid_mask
        return {
            "delta3": delta3,
            "gate32": gate32,
            "delta2": delta2,
            "p2_pc": p2_pc,
            "m2_pc": m2_pc,
        }


__all__ = ["P3P2CorrectionBridge"]
