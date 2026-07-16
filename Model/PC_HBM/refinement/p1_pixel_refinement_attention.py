"""Cross-grid P1 Pixel Refinement Attention (98 -> 28 references)."""

from __future__ import annotations

import math
from typing import Dict, Mapping

import torch
from torch import nn
import torch.nn.functional as F

from .boundary_query_head import BoundaryQueryHead1
from ..common.utils import (
    boundary_features_from_logits,
    finite_or_zero,
    gather_tokens,
    local_window_gather,
    masked_softmax,
    scatter_tokens,
)


class P1LocalStructuredPrior(nn.Module):
    """Interpretable 28-grid reference prior for each 98-grid query."""

    def __init__(self, in_dim: int = 6, hidden: int = 32) -> None:
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
        boundary2: torch.Tensor,
        gate2: torch.Tensor,
        distance: torch.Tensor,
        offset_magnitude: torch.Tensor,
        residual_terms: torch.Tensor,
    ) -> torch.Tensor:
        base = valid + boundary2 + gate2 - distance - 0.1 * offset_magnitude
        return base + torch.tanh(self.gamma) * self.residual(
            residual_terms
        ).squeeze(-1)


class P1PixelRefinementAttention(nn.Module):
    """Produce sparse raw residual/deformation/suppression maps at 98 x 98."""

    def __init__(
        self,
        p1_ch: int,
        dim: int = 128,
        window: int = 3,
        tau: float = 0.10,
        top_ratio: float = 0.05,
        min_tokens: int = 96,
        max_tokens: int | None = 384,
        detach_refs: bool = True,
        num_heads: int = 8,
        head_dim: int = 16,
        boundary_in_ch: int = 8,
    ) -> None:
        super().__init__()
        self.p1_ch = int(p1_ch)
        self.dim = int(dim)
        self.window = int(window)
        self.tau = float(tau)
        self.detach_refs = bool(detach_refs)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner = self.num_heads * self.head_dim
        self.boundary_in_ch = int(boundary_in_ch)
        if self.boundary_in_ch not in {8, 10}:
            raise ValueError("P1 boundary_in_ch must be 8 (legacy) or 10 (BGFBR)")
        if self.inner != self.dim:
            raise ValueError("num_heads * head_dim must equal dim")
        self.boundary_head = BoundaryQueryHead1(
            top_ratio=top_ratio,
            min_tokens=min_tokens,
            max_tokens=max_tokens,
            in_ch=self.boundary_in_ch,
        )
        self.query_encoder = nn.Conv2d(self.p1_ch, self.dim, 1, bias=False)
        self.ref_encoder = nn.Sequential(
            nn.Conv2d(self.dim + 1 + 1 + 2 + 1, self.dim, 1),
            nn.GroupNorm(8, self.dim),
            nn.GELU(),
            nn.Conv2d(self.dim, self.dim, 3, padding=1),
        )
        self.q_proj = nn.Linear(self.dim, self.inner)
        self.k_proj = nn.Linear(self.dim, self.inner)
        self.v_proj = nn.Linear(self.dim, self.inner)
        self.attn_out = nn.Linear(self.inner, self.dim)
        self.prior_residual = nn.Linear(self.dim, 1)
        self.structured_prior = P1LocalStructuredPrior(in_dim=6)
        self.g_head = nn.Linear(self.dim, 1)
        self.r_head = nn.Linear(self.dim, 1)
        self.o_head = nn.Linear(self.dim, 2)
        self.sup_head = nn.Linear(self.dim, 1)
        for head in (self.g_head, self.r_head, self.o_head, self.sup_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def build_boundary_input(
        self,
        z_main: torch.Tensor,
        p1_hw: tuple[int, int],
        p2_aux: Mapping[str, torch.Tensor],
        edge_context: torch.Tensor | None = None,
        dual_uncertainty: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_p1 = F.interpolate(
            z_main, size=p1_hw, mode="bilinear", align_corners=False
        )
        base = boundary_features_from_logits(z_p1)
        extras = [
            F.interpolate(
                p2_aux[name], size=p1_hw, mode="bilinear", align_corners=False
            )
            for name in ("B2_refined_map", "G2_refined_map", "valid2_map")
        ]
        boundary_input = torch.cat([base, *extras], dim=1)
        if self.boundary_in_ch == 10:
            zeros = torch.zeros_like(z_p1)
            edge = zeros if edge_context is None else edge_context
            dual = zeros if dual_uncertainty is None else dual_uncertainty
            edge = F.interpolate(edge, size=p1_hw, mode="bilinear", align_corners=False)
            dual = F.interpolate(dual, size=p1_hw, mode="bilinear", align_corners=False)
            boundary_input = torch.cat([boundary_input, edge, dual], dim=1)
        if boundary_input.size(1) != self.boundary_in_ch:
            raise RuntimeError(
                f"P1 boundary contract expected {self.boundary_in_ch} channels, "
                f"got {boundary_input.size(1)}"
            )
        return boundary_input

    def forward(
        self,
        p1: torch.Tensor,
        z_main: torch.Tensor,
        p2_aux: Mapping[str, torch.Tensor],
        edge_context: torch.Tensor | None = None,
        dual_uncertainty: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, _, height, width = p1.shape
        if (height, width) != (98, 98):
            raise ValueError(
                f"P1-PRA must run at 98x98, got {(height, width)}"
            )
        if z_main.shape != (batch_size, 1, height, width):
            raise ValueError("z_main must be [B,1,98,98] and match p1")
        boundary_input = self.build_boundary_input(
            z_main,
            (height, width),
            p2_aux,
            edge_context=edge_context,
            dual_uncertainty=dual_uncertainty,
        )
        boundary, indices = self.boundary_head(boundary_input)
        batch_ids = indices["batch_ids"]
        flat_indices = indices["flat_indices"]
        query_tokens = gather_tokens(
            self.query_encoder(p1), batch_ids, flat_indices
        )
        reference = torch.cat(
            [
                p2_aux["F2_ref_map"],
                p2_aux["B2_refined_map"],
                p2_aux["G2_refined_map"],
                p2_aux["O2_refined_map"],
                p2_aux["valid2_map"],
            ],
            dim=1,
        )
        if tuple(reference.shape[-2:]) != (28, 28):
            raise ValueError(
                "P1-PRA references must remain on the 28x28 p2 grid"
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
        local_valid2, _ = local_window_gather(
            p2_aux["valid2_map"],
            batch_ids,
            flat_indices,
            (height, width),
            tuple(reference_map.shape[-2:]),
            self.window,
        )
        candidate_valid = spatial_valid & (local_valid2[..., 0] > 0.5)
        query_valid = candidate_valid.any(dim=1)
        indices["query_valid"] = query_valid
        if query_tokens.numel() == 0:
            return self._empty(p1, boundary, reference_map, indices)

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
            p2_aux,
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
        gate_raw = self.g_head(features) * valid_float
        gate_value = torch.sigmoid(gate_raw) * valid_float
        residual_raw = self.r_head(features) * valid_float
        offset_raw = self.o_head(features) * valid_float
        suppress_raw = self.sup_head(features) * valid_float
        attention = attention_heads.mean(dim=1)

        def scatter(values: torch.Tensor, channels: int) -> torch.Tensor:
            return scatter_tokens(
                (batch_size, channels, height, width),
                batch_ids,
                flat_indices,
                values,
            )

        return {
            "G1_map": scatter(gate_value, 1),
            "G1_raw_map": scatter(gate_raw, 1),
            "R1_map": scatter(residual_raw, 1),
            "O1_map": scatter(offset_raw, 2),
            "R_sup_map": scatter(suppress_raw, 1),
            "valid1_map": scatter(valid_float, 1),
            "B1": boundary,
            "boundary_indices1": indices,
            "query_valid1": query_valid,
            "R2_map": reference_map,
            "attn1": attention,
            "prior1": prior,
        }

    def _empty(
        self,
        p1: torch.Tensor,
        boundary: torch.Tensor,
        reference_map: torch.Tensor,
        indices: dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        batch_size, _, height, width = p1.shape
        zeros1 = p1.new_zeros(batch_size, 1, height, width)
        return {
            "G1_map": zeros1,
            "G1_raw_map": zeros1.clone(),
            "R1_map": zeros1.clone(),
            "O1_map": p1.new_zeros(batch_size, 2, height, width),
            "R_sup_map": zeros1.clone(),
            "valid1_map": zeros1.clone(),
            "B1": boundary,
            "boundary_indices1": indices,
            "query_valid1": torch.empty(0, device=p1.device, dtype=torch.bool),
            "R2_map": reference_map,
            "attn1": p1.new_empty((0, self.window * self.window)),
            "prior1": p1.new_empty((0, self.window * self.window)),
        }

    def _structured_prior(
        self,
        p2_aux: Mapping[str, torch.Tensor],
        batch_ids: torch.Tensor,
        flat_indices: torch.Tensor,
        query_hw: tuple[int, int],
        ref_hw: tuple[int, int],
        candidate_valid: torch.Tensor,
        local_reference: torch.Tensor,
    ) -> torch.Tensor:
        def local(name: str) -> torch.Tensor:
            return local_window_gather(
                p2_aux[name],
                batch_ids,
                flat_indices,
                query_hw,
                ref_hw,
                self.window,
            )[0]

        boundary2 = local("B2_refined_map")[..., 0].clamp(0.0, 1.0)
        gate2 = local("G2_refined_map")[..., 0].clamp(0.0, 1.0)
        offset_magnitude = local("O2_refined_map").norm(dim=-1).clamp(0.0, 2.0)
        valid = candidate_valid.to(dtype=local_reference.dtype)
        distance = self._window_distance(
            local_reference.device, local_reference.dtype
        )[None].expand_as(valid)
        residual_terms = torch.stack(
            [valid, boundary2, gate2, distance, offset_magnitude, valid], dim=-1
        )
        prior = self.structured_prior(
            valid,
            boundary2,
            gate2,
            distance,
            offset_magnitude,
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
