"""Parent-child hypothesis cross-attention for DINO PC-HBM."""

from __future__ import annotations

import math

import torch
from torch import nn

from ..common.utils import finite_or_zero, masked_softmax, normalize


class PCHCA(nn.Module):
    """Attend ``[M,K,D]`` hypotheses with 8 x 16-dimensional heads.

    The route-conditioned residual modulation is zero-initialized. Fully
    invalid rows return their input query exactly and have zero attention.
    """

    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 8,
        head_dim: int = 16,
        tau: float = 0.10,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner = self.num_heads * self.head_dim
        if self.inner != self.dim:
            raise ValueError(
                "DINO PC-HBM requires num_heads * head_dim == dim, got "
                f"{self.num_heads} * {self.head_dim} != {self.dim}"
            )
        self.tau = float(tau)
        self.q_proj = nn.Linear(self.dim, self.inner)
        self.k_proj = nn.Linear(self.dim, self.inner)
        self.v_proj = nn.Linear(self.dim, self.inner)
        self.out_proj = nn.Linear(self.inner, self.dim)
        self.mod = nn.Linear(self.dim, self.dim * 3)
        nn.init.zeros_(self.mod.weight)
        nn.init.zeros_(self.mod.bias)

    def forward(
        self,
        q_state: torch.Tensor,
        h_tokens: torch.Tensor,
        prior_bias: torch.Tensor,
        route_context: torch.Tensor,
        mask: torch.Tensor | None = None,
        query_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if q_state.ndim != 2 or q_state.size(-1) != self.dim:
            raise ValueError(
                f"q_state must be [M,{self.dim}], got {tuple(q_state.shape)}"
            )
        if h_tokens.ndim != 3 or h_tokens.shape[0] != q_state.shape[0]:
            raise ValueError("h_tokens must be [M,K,D] and match q_state")
        if h_tokens.size(-1) != self.dim:
            raise ValueError(f"h_tokens last dimension must be {self.dim}")
        query_count, candidate_count, _ = h_tokens.shape
        if prior_bias.shape != (query_count, candidate_count):
            raise ValueError("prior_bias must be [M,K]")
        if route_context.shape != q_state.shape:
            raise ValueError("route_context must match q_state")
        if mask is None:
            mask = torch.ones(
                (query_count, candidate_count),
                device=h_tokens.device,
                dtype=torch.bool,
            )
        else:
            mask = mask.to(device=h_tokens.device, dtype=torch.bool)
            if mask.shape != (query_count, candidate_count):
                raise ValueError("mask must be [M,K]")
        if query_valid is None:
            query_valid = mask.any(dim=1)
        else:
            query_valid = query_valid.to(device=h_tokens.device, dtype=torch.bool)
            if query_valid.shape != (query_count,):
                raise ValueError("query_valid must be [M]")
        effective_valid = query_valid & mask.any(dim=1)
        mask = mask & effective_valid[:, None]
        if query_count == 0:
            return q_state, q_state.new_empty((0, candidate_count))

        query = self.q_proj(q_state).view(
            query_count, self.num_heads, self.head_dim
        )
        key = self.k_proj(h_tokens).view(
            query_count, candidate_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(h_tokens).view(
            query_count, candidate_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        logits = (query.unsqueeze(2) * key).sum(dim=-1)
        logits = logits / math.sqrt(self.head_dim) / max(self.tau, 1.0e-6)
        logits = logits + prior_bias.unsqueeze(1)
        attention_heads = masked_softmax(logits, mask[:, None, :], dim=-1)
        attended = (attention_heads.unsqueeze(-1) * value).sum(dim=2)
        attended = self.out_proj(attended.reshape(query_count, self.inner))

        shift, scale, strength = self.mod(route_context).chunk(3, dim=-1)
        modulated = attended * (1.0 + torch.tanh(scale)) + shift
        candidate_state = normalize(
            q_state + torch.tanh(strength) * modulated, dim=-1
        )
        output = torch.where(effective_valid[:, None], candidate_state, q_state)
        attention = attention_heads.mean(dim=1)
        attention = attention * effective_valid[:, None].to(attention.dtype)
        return finite_or_zero(output), finite_or_zero(attention)

