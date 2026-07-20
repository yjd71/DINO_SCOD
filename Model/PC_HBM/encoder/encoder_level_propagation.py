"""Detached-reference same-grid propagation from F3 to F2 and F2 to F1."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import nn

from Model.PC_HBM.common.utils import masked_softmax

from .encoder_feature_injector import _strength, _zero_linear


class SameGridLocalCrossAttention(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 8,
        window_size: int = 3,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError("num_heads must divide dim.")
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer.")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = dim // num_heads
        self.window_size = int(window_size)
        self.q = nn.Conv2d(dim, dim, 1, bias=False)
        self.k = nn.Conv2d(dim, dim, 1, bias=False)
        self.v = nn.Conv2d(dim, dim, 1, bias=False)
        self.out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(
        self,
        query_map: torch.Tensor,
        reference_map: torch.Tensor,
        evidence_map: torch.Tensor,
        valid_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if query_map.ndim != 4 or query_map.shape[1] != self.dim:
            raise ValueError(f"query_map must be [B,{self.dim},H,W].")
        if reference_map.shape != query_map.shape or evidence_map.shape != query_map.shape:
            raise ValueError("reference/evidence maps must match query_map.")
        if valid_map.shape != (query_map.shape[0], 1, *query_map.shape[-2:]):
            raise ValueError("valid_map must be [B,1,H,W].")
        batch, _, height, width = query_map.shape
        count = height * width
        neighbors = self.window_size * self.window_size
        q = self.q(query_map).flatten(2).transpose(1, 2)
        q = q.view(batch, count, self.num_heads, self.head_dim)
        source = reference_map + evidence_map
        padding = self.window_size // 2
        k = F.unfold(self.k(source), self.window_size, padding=padding)
        v = F.unfold(self.v(source), self.window_size, padding=padding)
        k = k.view(batch, self.num_heads, self.head_dim, neighbors, count)
        v = v.view(batch, self.num_heads, self.head_dim, neighbors, count)
        k = k.permute(0, 4, 1, 3, 2)
        v = v.permute(0, 4, 1, 3, 2)
        logits = (q.unsqueeze(3) * k).sum(dim=-1) / math.sqrt(self.head_dim)
        reference_valid = F.unfold(
            valid_map.to(dtype=query_map.dtype),
            self.window_size,
            padding=padding,
        )
        reference_valid = reference_valid.transpose(1, 2) > 0.5
        attention = masked_softmax(
            logits.float(), reference_valid[:, :, None, :], dim=-1
        ).to(dtype=q.dtype)
        attended = (attention.unsqueeze(-1) * v).sum(dim=3)
        attended = attended.reshape(batch, count, self.dim).transpose(1, 2)
        attended = self.out(attended.reshape(batch, self.dim, height, width))
        propagated_valid = reference_valid.any(dim=-1)
        propagated_valid_map = propagated_valid.view(batch, 1, height, width)
        state = query_map + attended * propagated_valid_map.to(attended.dtype)
        return state, attention.mean(dim=2), propagated_valid_map


@dataclass(frozen=True)
class EncoderPropagationOutput:
    f1_tokens: torch.Tensor
    f2_tokens: torch.Tensor
    f1_state: torch.Tensor
    f2_state: torch.Tensor
    f1_delta: torch.Tensor
    f2_delta: torch.Tensor
    f1_attention: torch.Tensor
    f2_attention: torch.Tensor
    valid1_map: torch.Tensor
    valid2_map: torch.Tensor


class EncoderLevelPropagation(nn.Module):
    def __init__(
        self,
        memory_dim: int = 128,
        encoder_dim: int = 768,
        *,
        num_heads: int = 8,
        window_size: int = 3,
        max_f2: float = 0.75,
        max_f1: float = 0.50,
        alpha_init: float = 1.0,
        detach_f3_refs: bool = True,
        detach_f2_refs: bool = True,
    ) -> None:
        super().__init__()
        self.memory_dim = int(memory_dim)
        self.encoder_dim = int(encoder_dim)
        self.max_f2 = float(max_f2)
        self.max_f1 = float(max_f1)
        self.detach_f3_refs = bool(detach_f3_refs)
        self.detach_f2_refs = bool(detach_f2_refs)
        self.f2_attention = SameGridLocalCrossAttention(
            memory_dim, num_heads, window_size
        )
        self.f1_attention = SameGridLocalCrossAttention(
            memory_dim, num_heads, window_size
        )
        self.restore2 = _zero_linear(memory_dim, encoder_dim)
        self.restore1 = _zero_linear(memory_dim, encoder_dim)
        self.alpha2 = nn.Parameter(torch.tensor(float(alpha_init)))
        self.alpha1 = nn.Parameter(torch.tensor(float(alpha_init)))

    def forward(
        self,
        *,
        f1_tokens: torch.Tensor,
        f2_tokens: torch.Tensor,
        e1_map: torch.Tensor,
        e2_map: torch.Tensor,
        corrected_f3_state: torch.Tensor,
        verified_f2_map: torch.Tensor,
        verified_f1_map: torch.Tensor,
        valid2_map: torch.Tensor,
        valid1_map: torch.Tensor,
        progress: float,
    ) -> EncoderPropagationOutput:
        self._validate(f1_tokens, e1_map, "F1")
        self._validate(f2_tokens, e2_map, "F2")
        expected_valid = (e1_map.shape[0], 1, *e1_map.shape[-2:])
        if valid1_map.shape != expected_valid or valid2_map.shape != expected_valid:
            raise ValueError("F1/F2 validity maps must be [B,1,H,W].")
        f3_reference = (
            corrected_f3_state.detach()
            if self.detach_f3_refs
            else corrected_f3_state
        )
        f2_state, f2_attention, propagated_valid2 = self.f2_attention(
            e2_map, f3_reference, verified_f2_map, valid2_map
        )
        f2_reference = f2_state.detach() if self.detach_f2_refs else f2_state
        f1_state, f1_attention, propagated_valid1 = self.f1_attention(
            e1_map, f2_reference, verified_f1_map, propagated_valid2
        )
        batch, _, height, width = f2_state.shape
        f2_memory_tokens = f2_state.flatten(2).transpose(1, 2)
        f1_memory_tokens = f1_state.flatten(2).transpose(1, 2)
        valid2_tokens = propagated_valid2.flatten(2).transpose(1, 2).to(f2_state.dtype)
        valid1_tokens = propagated_valid1.flatten(2).transpose(1, 2).to(f1_state.dtype)
        # Keep validity outside the affine restore projection so a learned
        # restore bias can never alter an invalid token.
        f2_delta = (
            _strength(self.max_f2, progress, self.alpha2)
            * valid2_tokens
            * self.restore2(f2_memory_tokens)
        )
        f1_delta = (
            _strength(self.max_f1, progress, self.alpha1)
            * valid1_tokens
            * self.restore1(f1_memory_tokens)
        )
        if f2_tokens.shape != (batch, height * width, self.encoder_dim):
            raise ValueError("F2 raw tokens do not match propagation maps.")
        if f1_tokens.shape != f2_tokens.shape:
            raise ValueError("F1/F2 raw token shapes must match.")
        return EncoderPropagationOutput(
            f1_tokens=f1_tokens + f1_delta,
            f2_tokens=f2_tokens + f2_delta,
            f1_state=f1_state,
            f2_state=f2_state,
            f1_delta=f1_delta,
            f2_delta=f2_delta,
            f1_attention=f1_attention,
            f2_attention=f2_attention,
            valid1_map=propagated_valid1,
            valid2_map=propagated_valid2,
        )

    def _validate(self, tokens: torch.Tensor, feature_map: torch.Tensor, level: str) -> None:
        if tokens.ndim != 3 or tokens.shape[-1] != self.encoder_dim:
            raise ValueError(f"{level} tokens must be [B,N,{self.encoder_dim}].")
        if feature_map.ndim != 4 or feature_map.shape[1] != self.memory_dim:
            raise ValueError(f"{level} map must be [B,{self.memory_dim},H,W].")


__all__ = [
    "EncoderLevelPropagation",
    "EncoderPropagationOutput",
    "SameGridLocalCrossAttention",
]
