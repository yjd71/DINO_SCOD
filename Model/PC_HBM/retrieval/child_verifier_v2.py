"""Masked child/geometry verification for retrieved parent hypotheses."""

from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn as nn

from ..common.utils import entropy_from_probs, js_divergence, masked_softmax, normalize_prob
from .geo_score_mlp import GeoScoreMLP
from .structured_prior_bias_net import StructuredPriorBiasNet


class ChildScoreMLP(nn.Module):
    def __init__(self, dim: int = 128, hidden: int = 128) -> None:
        super().__init__()
        self.dim = int(dim)
        self.net = nn.Sequential(
            nn.Linear(self.dim * 4, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), 1),
        )

    def forward(self, query: torch.Tensor, child_keys: torch.Tensor) -> torch.Tensor:
        if query.ndim != 2 or child_keys.ndim != 3:
            raise ValueError("query and child_keys must be [M,D] and [M,K,D]")
        if child_keys.shape[0] != query.shape[0] or child_keys.shape[2] != self.dim:
            raise ValueError("Child feature dimensions do not match")
        expanded = query.unsqueeze(1).expand_as(child_keys)
        features = torch.cat(
            (expanded, child_keys, (expanded - child_keys).abs(), expanded * child_keys),
            dim=-1,
        )
        return self.net(features).squeeze(-1)


class HypScoreNet(nn.Module):
    def __init__(self, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, int(hidden)),
            nn.GELU(),
            nn.Linear(int(hidden), 1),
        )

    def forward(
        self,
        parent_scores: torch.Tensor,
        child_scores: torch.Tensor,
        geometry_scores: torch.Tensor,
        prior_bias: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(
            torch.stack((parent_scores, child_scores, geometry_scores, prior_bias), dim=-1)
        ).squeeze(-1)


class ChildVerifierV2(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        value_dim: int = 8,
        geometry_dim: int = 6,
        tau: float = 0.10,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.value_dim = int(value_dim)
        self.geometry_dim = int(geometry_dim)
        self.tau = float(tau)
        if self.tau <= 0:
            raise ValueError("Child verification temperature must be positive")
        self.child_score = ChildScoreMLP(dim=self.dim)
        self.geo_score = GeoScoreMLP(geometry_dim=self.geometry_dim)
        self.prior = StructuredPriorBiasNet(
            value_dim=self.value_dim,
            geometry_dim=self.geometry_dim,
        )
        self.hyp_score = HypScoreNet()

    def forward(
        self,
        q_child: torch.Tensor,
        g2_query: torch.Tensor,
        parent_ret: Mapping[str, torch.Tensor],
        child_bank: Mapping[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        parent_keys = parent_ret["top_parent_keys"]
        parent_values = parent_ret["top_parent_values"]
        parent_geometry = parent_ret["top_parent_geo"]
        child_keys = child_bank["p2_child_keys"]
        child_geometry = child_bank["p2_child_geo"]
        expected_prefix = parent_keys.shape[:2]
        if child_keys.shape[:2] != expected_prefix or child_geometry.shape[:2] != expected_prefix:
            raise ValueError("Child bank shape must match retrieved [M,K] hypotheses")
        if q_child.shape != (parent_keys.size(0), self.dim):
            raise ValueError(f"q_child must be [M,{self.dim}], got {tuple(q_child.shape)}")
        if g2_query.shape != (parent_keys.size(0), self.geometry_dim):
            raise ValueError(
                f"g2_query must be [M,{self.geometry_dim}], got {tuple(g2_query.shape)}"
            )

        parent_valid = parent_ret.get("top_parent_valid")
        if parent_valid is None:
            parent_valid = parent_ret["top_child_ptrs"] >= 0
        child_valid = child_bank.get("child_valid")
        if child_valid is None:
            child_valid = torch.ones_like(parent_valid, dtype=torch.bool)
        valid = parent_valid.bool() & child_valid.to(device=parent_valid.device, dtype=torch.bool)
        query_valid = valid.any(dim=1)

        child_score = self.child_score(q_child, child_keys)
        geometry_score = self.geo_score(parent_geometry, child_geometry, g2_query)
        prior_bias = self.prior(
            parent_values,
            parent_geometry,
            child_geometry,
            child_score,
            geometry_score,
        )
        hypothesis_score = self.hyp_score(
            parent_ret["top_parent_scores"],
            child_score,
            geometry_score,
            prior_bias,
        )
        hypothesis_attention = masked_softmax(hypothesis_score / self.tau, valid, dim=1)
        pc_group = (
            hypothesis_attention.unsqueeze(-1) * parent_values[..., :4]
        ).sum(dim=1)
        pc_group = normalize_prob(pc_group, dim=1)
        contradiction_token = js_divergence(parent_ret["P3_group"], pc_group, dim=1).unsqueeze(1)
        contradiction_token = contradiction_token * query_valid.unsqueeze(1).to(contradiction_token.dtype)
        child_entropy = entropy_from_probs(hypothesis_attention, dim=1)

        child_positive = torch.sigmoid(child_score)
        contradiction = (
            parent_values[..., 5] * child_positive
            + parent_values[..., 4] * (1.0 - child_positive)
        ) * valid.to(child_score.dtype)
        zero = torch.zeros((), device=child_score.device, dtype=child_score.dtype)
        return {
            "S_child": torch.where(valid, child_score, zero),
            "S_geo": torch.where(valid, geometry_score, zero),
            "prior_bias": torch.where(valid, prior_bias, zero),
            "S_hyp": hypothesis_score.masked_fill(~valid, -1.0e4),
            "P_pc_group": pc_group,
            "C23_token": contradiction_token,
            "contradiction": contradiction,
            "child_entropy": child_entropy,
            "hyp_attn": hypothesis_attention,
            "K_child_top": child_keys,
            "G2_child_top": child_geometry,
            "top_parent_valid": valid,
            "query_valid": query_valid,
        }


__all__ = ["ChildScoreMLP", "ChildVerifierV2", "HypScoreNet"]
