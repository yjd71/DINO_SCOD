"""Learnable PC-HBM-Lite write-back without confidence gating or ramping."""

from __future__ import annotations

import torch
from torch import nn


class P3GatedResidual(nn.Module):
    """Project memory context and write it at valid selected P3 tokens."""

    def __init__(self, dim: int = 128, p3_ch: int = 128) -> None:
        super().__init__()
        self.dim = int(dim)
        self.p3_ch = int(p3_ch)
        self.out = nn.Linear(self.dim, self.p3_ch)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(
        self,
        p3: torch.Tensor,
        batch_ids: torch.Tensor,
        flat_indices: torch.Tensor,
        correction_token: torch.Tensor,
        query_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if p3.ndim != 4 or p3.size(1) != self.p3_ch:
            raise ValueError(
                f"p3 must be [B,{self.p3_ch},H,W], got {tuple(p3.shape)}"
            )
        query_count = batch_ids.numel()
        if flat_indices.shape != (query_count,):
            raise ValueError("flat_indices must align with batch_ids")
        if correction_token.shape != (query_count, self.dim):
            raise ValueError(f"correction_token must be [M,{self.dim}]")
        if query_valid.shape != (query_count,):
            raise ValueError("query_valid must be [M]")
        if batch_ids.dtype != torch.long or flat_indices.dtype != torch.long:
            raise TypeError("batch_ids and flat_indices must be torch.long")

        valid = query_valid.to(device=correction_token.device, dtype=torch.bool)
        safe_correction = torch.nan_to_num(
            correction_token, nan=0.0, posinf=0.0, neginf=0.0
        )
        delta_tokens = self.out(safe_correction)
        delta_tokens = delta_tokens * valid[:, None].to(dtype=delta_tokens.dtype)

        corrected = p3.clone()
        if query_count == 0:
            return corrected, delta_tokens
        height, width = p3.shape[-2:]
        if bool((batch_ids < 0).any()) or bool((batch_ids >= p3.size(0)).any()):
            raise IndexError("batch_ids contain an out-of-range index")
        if bool((flat_indices < 0).any()) or bool(
            (flat_indices >= height * width).any()
        ):
            raise IndexError("flat_indices contain an out-of-range index")
        row = torch.div(flat_indices, width, rounding_mode="floor")
        col = flat_indices.remainder(width)
        corrected[batch_ids, :, row, col] = (
            corrected[batch_ids, :, row, col] + delta_tokens
        )
        return corrected, delta_tokens
