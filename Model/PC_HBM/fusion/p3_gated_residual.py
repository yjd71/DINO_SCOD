"""Sparse, validity-aware p3 residual write-back."""

from __future__ import annotations

import torch
from torch import nn

from ..common.utils import add_tokens_to_map


class P3GatedResidual(nn.Module):
    """Project PC tokens to p3 channels and write them at query positions."""

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
        z3_token: torch.Tensor,
        gate: torch.Tensor | float,
        gate_pc: torch.Tensor,
        query_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_count = batch_ids.numel()
        if z3_token.shape != (query_count, self.dim):
            raise ValueError(f"z3_token must be [M,{self.dim}]")
        if not isinstance(gate, torch.Tensor):
            gate = z3_token.new_full((query_count, 1), float(gate))
        elif gate.ndim == 0:
            gate = gate.to(device=z3_token.device, dtype=z3_token.dtype).expand(
                query_count, 1
            )
        elif gate.ndim == 1:
            gate = gate[:, None]
        if gate_pc.ndim == 1:
            gate_pc = gate_pc[:, None]
        if gate.shape != (query_count, 1) or gate_pc.shape != (query_count, 1):
            raise ValueError("gate and gate_pc must be [M] or [M,1]")
        if query_valid is None:
            query_valid = torch.ones(
                query_count, device=z3_token.device, dtype=torch.bool
            )
        else:
            query_valid = query_valid.to(device=z3_token.device, dtype=torch.bool)
            if query_valid.shape != (query_count,):
                raise ValueError("query_valid must be [M]")
        delta = self.out(z3_token) * gate * gate_pc
        delta = delta * query_valid[:, None].to(dtype=delta.dtype)
        return add_tokens_to_map(p3, batch_ids, flat_indices, delta), delta
