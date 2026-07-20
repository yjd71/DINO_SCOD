"""PC-HCA fusion and structured gating for sparse encoder boundary queries."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from Model.PC_HBM.fusion.pc_hca import PCHCA


def _required_tensor(
    values: Mapping[str, Any], name: str, ndim: int
) -> torch.Tensor:
    value = values.get(name)
    if not torch.is_tensor(value) or value.ndim != ndim:
        raise ValueError(f"verification[{name!r}] must be a {ndim}-D tensor.")
    return value


def _entropy(probability: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    masked = probability * valid.to(probability.dtype)
    entropy = -(masked.clamp_min(1e-8) * masked.clamp_min(1e-8).log()).sum(dim=-1)
    count = valid.sum(dim=-1)
    normalizer = count.clamp_min(2).to(probability.dtype).log()
    return torch.where(count > 1, entropy / normalizer, torch.zeros_like(entropy))


def _masked_stats(
    scores: torch.Tensor,
    valid: torch.Tensor,
    *,
    include_std: bool,
) -> torch.Tensor:
    valid_float = valid.to(scores.dtype)
    count = valid_float.sum(dim=-1).clamp_min(1.0)
    mean = (scores * valid_float).sum(dim=-1) / count
    maximum = scores.masked_fill(~valid, float("-inf")).max(dim=-1).values
    maximum = torch.where(valid.any(dim=-1), maximum, torch.zeros_like(maximum))
    parts = [maximum, mean]
    if include_std:
        variance = ((scores - mean[:, None]).square() * valid_float).sum(dim=-1)
        parts.append(torch.sqrt(variance / count + 1e-8))
    return torch.stack(parts, dim=-1)


class _NAM(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(width, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class EncoderStructuredGate(nn.Module):
    """Encoder-only NAM gate; legacy decoder-side gate remains untouched."""

    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.gate_bias = nn.Parameter(torch.zeros(1))
        self.a_conf = nn.Parameter(torch.zeros(1))
        self.a_c23 = nn.Parameter(torch.zeros(1))
        self.a_uncertainty = nn.Parameter(torch.zeros(1))
        self.a_parent_entropy = nn.Parameter(torch.zeros(1))
        self.a_child_entropy = nn.Parameter(torch.zeros(1))
        self.a_retrieval = nn.Parameter(torch.zeros(1))
        self.semantic_nam = _NAM(3)
        self.detail_nam = _NAM(2)
        self.geometry_nam = _NAM(2)
        # route confidence, contradiction, uncertainty, two entropies,
        # parent score max/mean, 3+2+2 NAM statistics, boundary confidence.
        self.gate_interaction_mlp = nn.Sequential(
            nn.Linear(15, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), 1),
        )
        self.gamma_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        *,
        route_confidence: torch.Tensor,
        contradiction: torch.Tensor,
        uncertainty: torch.Tensor,
        parent_entropy: torch.Tensor,
        child_entropy: torch.Tensor,
        parent_scores: torch.Tensor,
        semantic_scores: torch.Tensor,
        detail_scores: torch.Tensor,
        geometry_scores: torch.Tensor,
        boundary_confidence: torch.Tensor,
        valid: torch.Tensor,
        query_valid: torch.Tensor,
    ) -> torch.Tensor:
        rows = valid.shape[0]

        def column(value: torch.Tensor, name: str) -> torch.Tensor:
            if value.ndim == 1:
                value = value[:, None]
            if value.shape != (rows, 1):
                raise ValueError(f"{name} must be [M] or [M,1].")
            return value

        route_confidence = column(route_confidence, "route_confidence")
        contradiction = column(contradiction, "contradiction")
        uncertainty = column(uncertainty, "uncertainty")
        boundary_confidence = column(boundary_confidence, "boundary_confidence")
        if any(value.shape != valid.shape for value in (
            parent_scores,
            semantic_scores,
            detail_scores,
            geometry_scores,
        )):
            raise ValueError("Structured gate score tensors must align with valid [M,K].")
        semantic_stats = _masked_stats(semantic_scores, valid, include_std=True)
        detail_stats = _masked_stats(detail_scores, valid, include_std=False)
        geometry_stats = _masked_stats(geometry_scores, valid, include_std=False)
        parent_stats = _masked_stats(parent_scores, valid, include_std=False)
        parent_entropy = column(parent_entropy, "parent_entropy")
        child_entropy = column(child_entropy, "child_entropy")
        gate_base = (
            self.gate_bias
            + F.softplus(self.a_conf) * route_confidence
            - F.softplus(self.a_c23) * contradiction
            - F.softplus(self.a_uncertainty) * uncertainty
            - F.softplus(self.a_parent_entropy) * parent_entropy
            - F.softplus(self.a_child_entropy) * child_entropy
            + F.softplus(self.a_retrieval) * parent_stats[:, :1]
            + self.semantic_nam(semantic_stats)
            + self.detail_nam(detail_stats)
            + self.geometry_nam(geometry_stats)
        )
        interaction = torch.cat(
            (
                route_confidence,
                contradiction,
                uncertainty,
                parent_entropy,
                child_entropy,
                parent_stats,
                semantic_stats,
                detail_stats,
                geometry_stats,
                boundary_confidence,
            ),
            dim=-1,
        )
        residual = self.gate_interaction_mlp(interaction)
        gate = torch.sigmoid(gate_base + torch.tanh(self.gamma_gate) * residual)
        return gate * query_valid[:, None].to(gate.dtype)


def scatter_query_tokens(
    tokens: torch.Tensor,
    batch_ids: torch.Tensor,
    flat_indices: torch.Tensor,
    *,
    batch_size: int,
    token_size: int,
) -> torch.Tensor:
    if tokens.ndim != 2:
        raise ValueError("tokens must be [M,C].")
    if batch_ids.shape != flat_indices.shape or batch_ids.ndim != 1:
        raise ValueError("batch_ids and flat_indices must be aligned [M].")
    if tokens.shape[0] != batch_ids.numel():
        raise ValueError("Sparse tokens and indices do not align.")
    count = token_size * token_size
    linear = batch_ids.long() * count + flat_indices.long()
    output = tokens.new_zeros(batch_size * count, tokens.shape[1])
    denominator = tokens.new_zeros(batch_size * count, 1)
    output.index_add_(0, linear, tokens)
    denominator.index_add_(0, linear, tokens.new_ones(tokens.shape[0], 1))
    output = output / denominator.clamp_min(1.0)
    return (
        output.view(batch_size, token_size, token_size, tokens.shape[1])
        .permute(0, 3, 1, 2)
        .contiguous()
    )


@dataclass(frozen=True)
class EncoderRouteContextOutput:
    z3_tokens: torch.Tensor
    hca_attention: torch.Tensor
    verified_f3_map: torch.Tensor
    verified_f2_map: torch.Tensor
    verified_f1_map: torch.Tensor
    gate_map: torch.Tensor
    valid3_map: torch.Tensor
    valid2_map: torch.Tensor
    valid1_map: torch.Tensor
    c23_map: torch.Tensor
    semantic_support_map: torch.Tensor
    detail_support_map: torch.Tensor
    geometry_support_map: torch.Tensor


class EncoderRouteContextAdapter(nn.Module):
    """Build hypothesis tokens, run PC-HCA, and scatter structured evidence."""

    def __init__(
        self,
        dim: int = 128,
        *,
        num_heads: int = 8,
        token_size: int = 28,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.token_size = int(token_size)
        hypothesis_dim = dim * 3 + 8 + 6 + 6
        self.hypothesis_encoder = nn.Sequential(
            nn.Linear(hypothesis_dim, 256),
            nn.GELU(),
            nn.Linear(256, dim),
            nn.LayerNorm(dim),
        )
        self.hca = PCHCA(dim=dim, num_heads=num_heads, head_dim=dim // num_heads)
        self.gate = EncoderStructuredGate()

    def forward(
        self,
        *,
        q3: torch.Tensor,
        verification: Mapping[str, Any],
        route_context: torch.Tensor,
        route_confidence: torch.Tensor,
        uncertainty: torch.Tensor,
        boundary_confidence: torch.Tensor | None = None,
        batch_ids: torch.Tensor,
        flat_indices: torch.Tensor,
        batch_size: int,
    ) -> EncoderRouteContextOutput:
        if q3.ndim != 2 or q3.shape[-1] != self.dim:
            raise ValueError(f"q3 must be [M,{self.dim}].")
        rows = q3.shape[0]
        if route_context.shape != q3.shape:
            raise ValueError("route_context must match q3.")
        if route_confidence.shape not in {(rows,), (rows, 1)}:
            raise ValueError("route_confidence must be [M] or [M,1].")
        if uncertainty.shape not in {(rows,), (rows, 1)}:
            raise ValueError("uncertainty must be [M] or [M,1].")
        if boundary_confidence is None:
            boundary_confidence = 1.0 - uncertainty
        if boundary_confidence.shape not in {(rows,), (rows, 1)}:
            raise ValueError("boundary_confidence must be [M] or [M,1].")

        parent = _required_tensor(verification, "top_parent_keys", 3).to(q3)
        semantic = _required_tensor(
            verification, "top_child_semantic_keys", 3
        ).to(q3)
        detail = _required_tensor(verification, "top_child_detail_keys", 3).to(q3)
        values = _required_tensor(verification, "top_parent_values", 3).to(q3)
        geometry = _required_tensor(verification, "top_parent_geometry", 3).to(q3)
        valid = _required_tensor(verification, "top_parent_valid", 2).bool()
        scalars = torch.stack(
            (
                _required_tensor(verification, "top_parent_scores", 2).to(q3),
                _required_tensor(verification, "S_semantic", 2).to(q3),
                _required_tensor(verification, "S_detail", 2).to(q3),
                _required_tensor(verification, "S_geometry", 2).to(q3),
                _required_tensor(verification, "prior_bias", 2).to(q3),
                _required_tensor(verification, "contradiction", 2).to(q3),
            ),
            dim=-1,
        )
        hypothesis = self.hypothesis_encoder(
            torch.cat((parent, semantic, detail, values, geometry, scalars), dim=-1)
        )
        query_valid = valid.any(dim=-1)
        z3, hca_attention = self.hca(
            q3,
            hypothesis,
            _required_tensor(verification, "prior_bias", 2).to(q3),
            route_context,
            mask=valid,
            query_valid=query_valid,
        )
        parent_attention = _required_tensor(
            verification, "parent_attention", 2
        ).to(q3)
        child_attention = _required_tensor(
            verification, "hypothesis_attention", 2
        ).to(q3)
        c23 = _required_tensor(verification, "contradiction_token", 2).to(q3)
        gate = self.gate(
            route_confidence=route_confidence,
            contradiction=c23,
            uncertainty=uncertainty,
            parent_entropy=_entropy(parent_attention, valid),
            child_entropy=_entropy(child_attention, valid),
            parent_scores=_required_tensor(
                verification, "top_parent_scores", 2
            ).to(q3),
            semantic_scores=_required_tensor(
                verification, "S_semantic", 2
            ).to(q3),
            detail_scores=_required_tensor(verification, "S_detail", 2).to(q3),
            geometry_scores=_required_tensor(
                verification, "S_geometry", 2
            ).to(q3),
            boundary_confidence=boundary_confidence,
            valid=valid,
            query_valid=query_valid,
        )
        valid_column = query_valid[:, None].to(q3.dtype)
        z3 = z3 * valid_column
        semantic_evidence = _required_tensor(
            verification, "semantic_evidence", 2
        ).to(q3) * valid_column
        detail_evidence = _required_tensor(
            verification, "detail_evidence", 2
        ).to(q3) * valid_column
        semantic_support = _required_tensor(
            verification, "semantic_support", 2
        ).to(q3)
        detail_support = _required_tensor(verification, "detail_support", 2).to(q3)
        geometry_support = _required_tensor(
            verification, "geometry_support", 2
        ).to(q3)

        def scatter(value: torch.Tensor) -> torch.Tensor:
            return scatter_query_tokens(
                value,
                batch_ids,
                flat_indices,
                batch_size=batch_size,
                token_size=self.token_size,
            )

        valid_map = scatter(valid_column).bool()
        return EncoderRouteContextOutput(
            z3_tokens=z3,
            hca_attention=hca_attention,
            verified_f3_map=scatter(z3),
            verified_f2_map=scatter(semantic_evidence),
            verified_f1_map=scatter(detail_evidence),
            gate_map=scatter(gate),
            valid3_map=valid_map,
            valid2_map=valid_map,
            valid1_map=valid_map,
            c23_map=scatter(c23),
            semantic_support_map=scatter(semantic_support),
            detail_support_map=scatter(detail_support),
            geometry_support_map=scatter(geometry_support),
        )


__all__ = [
    "EncoderRouteContextAdapter",
    "EncoderRouteContextOutput",
    "EncoderStructuredGate",
    "scatter_query_tokens",
]
