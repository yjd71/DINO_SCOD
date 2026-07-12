"""Encode retrieved parent-child hypotheses into compact DINO tokens."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

from ..common.utils import normalize


def _candidate_validity(
    parent_ret: Mapping[str, torch.Tensor],
    reference: torch.Tensor,
    top_parent_valid: torch.Tensor | None,
    query_valid: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return candidate/query masks on the reference tensor's device."""

    if top_parent_valid is None:
        top_parent_valid = parent_ret.get("top_parent_valid")
    if top_parent_valid is None:
        top_parent_valid = torch.ones(
            reference.shape[:2], device=reference.device, dtype=torch.bool
        )
    else:
        top_parent_valid = top_parent_valid.to(
            device=reference.device, dtype=torch.bool
        )
    if tuple(top_parent_valid.shape) != tuple(reference.shape[:2]):
        raise ValueError(
            "top_parent_valid must match the [M,K] candidate axes, got "
            f"{tuple(top_parent_valid.shape)} for {tuple(reference.shape)}"
        )
    if query_valid is None:
        query_valid = top_parent_valid.any(dim=1)
    else:
        query_valid = query_valid.to(device=reference.device, dtype=torch.bool)
        if tuple(query_valid.shape) != (reference.size(0),):
            raise ValueError(
                f"query_valid must be [M], got {tuple(query_valid.shape)}"
            )
    candidate_valid = top_parent_valid & query_valid[:, None]
    return candidate_valid, candidate_valid.any(dim=1)


class HypothesisTokenBuilder(nn.Module):
    """Build normalized ``[M,K,D]`` hypothesis tokens.

    Invalid padded parents never leak their sentinel contents into a later
    attention layer: their complete output token is an exact zero.
    """

    def __init__(
        self,
        dim: int = 128,
        value_dim: int = 8,
        geometry_dim: int = 6,
        hidden: int | None = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.value_dim = int(value_dim)
        self.geometry_dim = int(geometry_dim)
        hidden_dim = self.dim if hidden is None else int(hidden)
        self.region_embed = nn.Embedding(4, self.dim)
        input_dim = (
            self.dim * 3
            + self.value_dim
            + self.geometry_dim * 2
            + 4
        )
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.dim),
        )

    def forward(
        self,
        parent_ret: Mapping[str, torch.Tensor],
        child_ver: Mapping[str, torch.Tensor],
        top_parent_valid: torch.Tensor | None = None,
        query_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        parent_keys = parent_ret["top_parent_keys"]
        values = parent_ret["top_parent_values"]
        child_keys = child_ver["K_child_top"]
        parent_geometry = parent_ret["top_parent_geo"]
        child_geometry = child_ver["G2_child_top"]
        expected_prefix = tuple(parent_keys.shape[:2])
        tensors = {
            "top_parent_values": values,
            "K_child_top": child_keys,
            "top_parent_geo": parent_geometry,
            "G2_child_top": child_geometry,
        }
        for name, tensor in tensors.items():
            if tuple(tensor.shape[:2]) != expected_prefix:
                raise ValueError(
                    f"{name} candidate axes {tuple(tensor.shape[:2])} do not "
                    f"match parent keys {expected_prefix}"
                )
        valid, _ = _candidate_validity(
            parent_ret, parent_keys, top_parent_valid, query_valid
        )
        child_valid = child_ver.get("top_parent_valid")
        if child_valid is not None:
            child_valid = child_valid.to(device=valid.device, dtype=torch.bool)
            if child_valid.shape != valid.shape:
                raise ValueError("child top_parent_valid must be [M,K]")
            valid = valid & child_valid
        region_id = values[..., :4].argmax(dim=-1).clamp(0, 3)
        region = self.region_embed(region_id)
        scores = torch.stack(
            [
                parent_ret["top_parent_scores"],
                child_ver["S_child"],
                child_ver["S_geo"],
                child_ver["prior_bias"],
            ],
            dim=-1,
        )
        features = torch.cat(
            [
                parent_keys,
                child_keys,
                values,
                parent_geometry,
                child_geometry,
                scores,
                region,
            ],
            dim=-1,
        )
        tokens = normalize(self.net(features), dim=-1)
        return tokens * valid.unsqueeze(-1).to(dtype=tokens.dtype)
