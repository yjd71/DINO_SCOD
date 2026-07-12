"""Same-grid P2 Boundary Retarget Attention (28 -> 28)."""

from __future__ import annotations

import math
from typing import Dict, Mapping

import torch
from torch import nn

from .boundary_query_head import BoundaryQueryHead2
from ..common.utils import (
    add_tokens_to_map,
    boundary_features_from_logits,
    finite_or_zero,
    gather_tokens,
    local_window_gather,
    masked_softmax,
    scatter_tokens,
)


class P2LocalStructuredPrior(nn.Module):
    """Interpretable local prior plus a gated learned residual."""

    def __init__(self, in_dim: int = 8, hidden: int = 32) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.residual = nn.Sequential(
            nn.Linear(int(in_dim), int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), 1),
        )

    def forward(
        self,
        valid: torch.Tensor,
        gate: torch.Tensor,
        c23: torch.Tensor,
        m_pc: torch.Tensor,
        dist: torch.Tensor,
        offset_mag: torch.Tensor,
        reliability: torch.Tensor,
        residual_terms: torch.Tensor,
    ) -> torch.Tensor:
        base = (
            valid
            + gate
            - c23
            + 0.25 * m_pc
            + 0.25 * reliability
            - dist
            - 0.1 * offset_mag
        )
        return base + torch.tanh(self.gamma) * self.residual(
            residual_terms
        ).squeeze(-1)


class P2BoundaryRetargetAttention(nn.Module):
    """Retarget verified p3 references to p2 boundary tokens."""

    def __init__(
        self,
        p2_ch: int,
        dim: int = 128,
        window: int = 3,
        tau: float = 0.10,
        top_ratio: float = 0.20,
        min_tokens: int = 32,
        max_tokens: int | None = 128,
        detach_refs: bool = True,
        num_heads: int = 8,
        head_dim: int = 16,
    ) -> None:
        super().__init__()
        self.p2_ch = int(p2_ch)
        self.dim = int(dim)
        self.window = int(window)
        self.tau = float(tau)
        self.detach_refs = bool(detach_refs)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner = self.num_heads * self.head_dim
        if self.inner != self.dim:
            raise ValueError("num_heads * head_dim must equal dim")
        self.boundary_head = BoundaryQueryHead2(
            top_ratio=top_ratio,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
        )
        self.query_encoder = nn.Conv2d(self.p2_ch, self.dim, 1, bias=False)
        ref_channels = self.dim + 8 + 6 + 1 + 1 + 1 + 2 + 1
        self.ref_encoder = nn.Sequential(
            nn.Conv2d(ref_channels, self.dim, 1),
            nn.GroupNorm(8, self.dim),
            nn.GELU(),
            nn.Conv2d(self.dim, self.dim, 3, padding=1),
        )
        self.q_proj = nn.Linear(self.dim, self.inner)
        self.k_proj = nn.Linear(self.dim, self.inner)
        self.v_proj = nn.Linear(self.dim, self.inner)
        self.attn_out = nn.Linear(self.inner, self.dim)
        self.prior_residual = nn.Linear(self.dim, 1)
        self.structured_prior = P2LocalStructuredPrior(in_dim=8)
        self.restore = nn.Linear(self.dim, self.p2_ch)
        self.b_head = nn.Linear(self.dim, 1)
        self.g_head = nn.Linear(self.dim, 1)
        self.o_head = nn.Linear(self.dim, 2)
        self.gate = nn.Linear(self.dim, 1)
        nn.init.zeros_(self.restore.weight)
        nn.init.zeros_(self.restore.bias)
        nn.init.zeros_(self.o_head.weight)
        nn.init.zeros_(self.o_head.bias)

    def build_boundary_input(
        self, prob2: torch.Tensor, pc_maps: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        if prob2.ndim != 4 or prob2.size(1) != 1:
            raise ValueError("prob2 must be [B,1,H,W]")
        dtype_epsilon = (
            torch.finfo(prob2.dtype).eps if prob2.is_floating_point() else 1.0e-6
        )
        epsilon = max(1.0e-6, float(dtype_epsilon))
        logits = torch.logit(prob2.clamp(epsilon, 1.0 - epsilon))
        base = boundary_features_from_logits(logits)
        target = prob2.shape[-2:]
        extras = [
            torch.nn.functional.interpolate(
                pc_maps[name], size=target, mode="bilinear", align_corners=False
            )
            for name in ("gate_pc_map", "C23_map", "M_pc_map")
        ]
        return torch.cat([base, *extras], dim=1)

    def forward(
        self,
        p2: torch.Tensor,
        prob2: torch.Tensor,
        pc_maps: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        batch_size, _, height, width = p2.shape
        if (height, width) != (28, 28):
            raise ValueError(
                f"P2-BRA must run on the 28x28 grid, got {(height, width)}"
            )
        if prob2.shape != (batch_size, 1, height, width):
            raise ValueError("P2-BRA is same-grid: prob2 must match p2 spatially")
        boundary_input = self.build_boundary_input(prob2, pc_maps)
        boundary, indices = self.boundary_head(boundary_input)
        batch_ids = indices["batch_ids"]
        flat_indices = indices["flat_indices"]
        query_map = self.query_encoder(p2)
        query_tokens = gather_tokens(query_map, batch_ids, flat_indices)
        reference = torch.cat(
            [
                pc_maps["Z3_map"],
                pc_maps["E_attn_map"],
                pc_maps["G_attn_map"],
                pc_maps["M_pc_map"],
                pc_maps["gate_pc_map"],
                pc_maps["C23_map"],
                pc_maps["O_pc_map"],
                pc_maps["valid3_map"],
            ],
            dim=1,
        )
        if self.detach_refs:
            reference = reference.detach()
        reference_map = self.ref_encoder(reference)
        local_reference, spatial_valid = local_window_gather(
            reference_map,
            batch_ids,
            flat_indices,
            (height, width),
            tuple(reference_map.shape[-2:]),
            self.window,
        )
        local_valid3, _ = local_window_gather(
            pc_maps["valid3_map"],
            batch_ids,
            flat_indices,
            (height, width),
            tuple(reference_map.shape[-2:]),
            self.window,
        )
        candidate_valid = spatial_valid & (local_valid3[..., 0] > 0.5)
        query_valid = candidate_valid.any(dim=1)
        indices["query_valid"] = query_valid
        if query_tokens.numel() == 0:
            return self._empty(p2, boundary, reference_map, indices)

        query_count, candidate_count, _ = local_reference.shape
        query = self.q_proj(query_tokens).view(
            query_count, self.num_heads, self.head_dim
        )
        key = self.k_proj(local_reference).view(
            query_count, candidate_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(local_reference).view(
            query_count, candidate_count, self.num_heads, self.head_dim
        ).transpose(1, 2)
        logits = (query.unsqueeze(2) * key).sum(dim=-1)
        logits = logits / math.sqrt(self.head_dim) / max(self.tau, 1.0e-6)
        prior = self._structured_prior(
            pc_maps,
            batch_ids,
            flat_indices,
            (height, width),
            tuple(reference_map.shape[-2:]),
            candidate_valid,
            local_reference,
        )
        logits = logits + prior[:, None, :] + self.prior_residual(
            local_reference
        ).squeeze(-1)[:, None, :]
        attention_heads = masked_softmax(
            logits, candidate_valid[:, None, :], dim=-1
        )
        attended = (attention_heads.unsqueeze(-1) * value).sum(dim=2)
        features = finite_or_zero(
            self.attn_out(attended.reshape(query_count, self.inner))
        )
        valid_float = query_valid[:, None].to(dtype=features.dtype)
        features = features * valid_float
        gate = torch.sigmoid(self.gate(features)) * valid_float
        correction = self.restore(features) * gate
        refined = add_tokens_to_map(
            p2, batch_ids, flat_indices, correction
        )
        b_value = torch.sigmoid(self.b_head(features)) * valid_float
        g_value = torch.sigmoid(self.g_head(features)) * valid_float
        o_value = self.o_head(features) * valid_float
        ones = valid_float
        attention = attention_heads.mean(dim=1)
        return {
            "p2_refined": refined,
            "F2_ref_map": scatter_tokens(
                (batch_size, self.dim, height, width),
                batch_ids,
                flat_indices,
                features,
            ),
            "B2_refined_map": scatter_tokens(
                (batch_size, 1, height, width), batch_ids, flat_indices, b_value
            ),
            "G2_refined_map": scatter_tokens(
                (batch_size, 1, height, width), batch_ids, flat_indices, g_value
            ),
            "O2_refined_map": scatter_tokens(
                (batch_size, 2, height, width), batch_ids, flat_indices, o_value
            ),
            "valid2_map": scatter_tokens(
                (batch_size, 1, height, width), batch_ids, flat_indices, ones
            ),
            "B2": boundary,
            "boundary_indices2": indices,
            "query_valid2": query_valid,
            "R3_map": reference_map,
            "attn2": attention,
            "prior2": prior,
        }

    def _empty(
        self,
        p2: torch.Tensor,
        boundary: torch.Tensor,
        reference_map: torch.Tensor,
        indices: dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        batch_size, _, height, width = p2.shape
        return {
            "p2_refined": p2,
            "F2_ref_map": p2.new_zeros(batch_size, self.dim, height, width),
            "B2_refined_map": p2.new_zeros(batch_size, 1, height, width),
            "G2_refined_map": p2.new_zeros(batch_size, 1, height, width),
            "O2_refined_map": p2.new_zeros(batch_size, 2, height, width),
            "valid2_map": p2.new_zeros(batch_size, 1, height, width),
            "B2": boundary,
            "boundary_indices2": indices,
            "query_valid2": torch.empty(0, device=p2.device, dtype=torch.bool),
            "R3_map": reference_map,
            "attn2": p2.new_empty((0, self.window * self.window)),
            "prior2": p2.new_empty((0, self.window * self.window)),
        }

    def _structured_prior(
        self,
        pc_maps: Mapping[str, torch.Tensor],
        batch_ids: torch.Tensor,
        flat_indices: torch.Tensor,
        query_hw: tuple[int, int],
        ref_hw: tuple[int, int],
        candidate_valid: torch.Tensor,
        local_reference: torch.Tensor,
    ) -> torch.Tensor:
        def local(name: str) -> torch.Tensor:
            return local_window_gather(
                pc_maps[name],
                batch_ids,
                flat_indices,
                query_hw,
                ref_hw,
                self.window,
            )[0]

        gate = local("gate_pc_map")[..., 0].clamp(0.0, 1.0)
        c23 = local("C23_map")[..., 0].clamp(0.0, 1.0)
        mask_pc = local("M_pc_map")[..., 0].clamp(-1.0, 1.0)
        offset_magnitude = local("O_pc_map").norm(dim=-1).clamp(0.0, 2.0)
        evidence = local("E_attn_map")
        valid = candidate_valid.to(dtype=local_reference.dtype)
        reliability = (
            evidence[..., 7].clamp(0.0, 1.0)
            if evidence.size(-1) > 7
            else valid
        )
        distance = self._window_distance(
            local_reference.device, local_reference.dtype
        )[None].expand_as(valid)
        residual_terms = torch.stack(
            [
                valid,
                gate,
                c23,
                mask_pc,
                distance,
                offset_magnitude,
                reliability,
                valid,
            ],
            dim=-1,
        )
        prior = self.structured_prior(
            valid,
            gate,
            c23,
            mask_pc,
            distance,
            offset_magnitude,
            reliability,
            residual_terms,
        )
        return prior.masked_fill(~candidate_valid, -1.0e4)

    def _window_distance(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        radius = self.window // 2
        denominator = max(float(radius), 1.0)
        values = [
            math.sqrt(dx * dx + dy * dy) / denominator
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
        ]
        return torch.tensor(values, device=device, dtype=dtype)
