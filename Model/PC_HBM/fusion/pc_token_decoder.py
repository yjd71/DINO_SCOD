"""Decode attended hypotheses into sparse correction tokens."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from ..common.utils import normalize_prob


class PCTokenDecoder(nn.Module):
    """Produce evidence, geometry, mask and offset tokens for map scatter."""

    def __init__(
        self,
        dim: int = 128,
        value_dim: int = 8,
        geometry_dim: int = 6,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.value_dim = int(value_dim)
        self.geometry_dim = int(geometry_dim)
        self.evidence_residual = nn.Linear(self.dim, self.value_dim)
        self.mask_residual = nn.Linear(self.dim, 1)
        self.offset = nn.Linear(
            self.dim + self.geometry_dim * 2 + 1, 2
        )
        for layer in (self.evidence_residual, self.mask_residual, self.offset):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(
        self,
        q3_new: torch.Tensor,
        attn: torch.Tensor,
        parent_ret: Mapping[str, torch.Tensor],
        child_ver: Mapping[str, torch.Tensor],
        top_parent_valid: torch.Tensor | None = None,
        query_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if q3_new.ndim != 2 or q3_new.size(1) != self.dim:
            raise ValueError(
                f"q3_new must be [M,{self.dim}], got {tuple(q3_new.shape)}"
            )
        query_count = q3_new.size(0)
        if attn.ndim != 2 or attn.size(0) != query_count:
            raise ValueError("attn must be [M,K]")
        candidate_count = attn.size(1)
        if top_parent_valid is None:
            top_parent_valid = parent_ret.get("top_parent_valid")
        if top_parent_valid is None:
            top_parent_valid = torch.ones_like(attn, dtype=torch.bool)
        else:
            top_parent_valid = top_parent_valid.to(
                device=attn.device, dtype=torch.bool
            )
        if top_parent_valid.shape != (query_count, candidate_count):
            raise ValueError("top_parent_valid must be [M,K]")
        if query_valid is None:
            query_valid = top_parent_valid.any(dim=1)
        else:
            query_valid = query_valid.to(device=attn.device, dtype=torch.bool)
            if query_valid.shape != (query_count,):
                raise ValueError("query_valid must be [M]")
        query_valid = query_valid & top_parent_valid.any(dim=1)
        valid = top_parent_valid & query_valid[:, None]
        child_valid = child_ver.get("top_parent_valid")
        if child_valid is not None:
            child_valid = child_valid.to(device=valid.device, dtype=torch.bool)
            if child_valid.shape != valid.shape:
                raise ValueError("child top_parent_valid must be [M,K]")
            valid = valid & child_valid
            query_valid = query_valid & valid.any(dim=1)
        weights = normalize_prob(
            attn * valid.to(dtype=attn.dtype), dim=1
        )

        parent_values = parent_ret["top_parent_values"]
        parent_geometry = parent_ret["top_parent_geo"]
        child_geometry = child_ver["G2_child_top"]
        if parent_values.shape != (query_count, candidate_count, self.value_dim):
            raise ValueError(
                "top_parent_values must be "
                f"[M,K,{self.value_dim}], got {tuple(parent_values.shape)}"
            )
        for name, tensor in {
            "top_parent_geo": parent_geometry,
            "G2_child_top": child_geometry,
        }.items():
            if tensor.shape != (
                query_count,
                candidate_count,
                self.geometry_dim,
            ):
                raise ValueError(
                    f"{name} must be [M,K,{self.geometry_dim}], got "
                    f"{tuple(tensor.shape)}"
                )

        evidence = (weights.unsqueeze(-1) * parent_values).sum(dim=1)
        parent_geo = (weights.unsqueeze(-1) * parent_geometry).sum(dim=1)
        child_geo = (weights.unsqueeze(-1) * child_geometry).sum(dim=1)
        evidence = evidence + 0.1 * self.evidence_residual(q3_new)
        foreground = evidence[:, 0] + evidence[:, 1]
        background = evidence[:, 2] + evidence[:, 3]
        mask_evidence = foreground - background
        mask_residual = 0.1 * torch.tanh(
            self.mask_residual(q3_new)
        ).squeeze(-1)
        mask_token = (mask_evidence + mask_residual).clamp(-1.0, 1.0)
        offset_features = torch.cat(
            [q3_new, parent_geo, child_geo, mask_token[:, None]], dim=-1
        )
        offset = torch.tanh(self.offset(offset_features))

        valid_float = query_valid[:, None].to(dtype=q3_new.dtype)
        return {
            "E_attn": evidence * valid_float,
            "G_attn": parent_geo * valid_float,
            "G_child_attn": child_geo * valid_float,
            "M_pc_token": mask_token[:, None] * valid_float,
            "M_pc_evidence": mask_evidence[:, None] * valid_float,
            "M_pc_residual": mask_residual[:, None] * valid_float,
            "O_pc_token": offset * valid_float,
            "Z3_token": q3_new * valid_float,
            "valid_token": query_valid[:, None],
        }
