"""Scatter valid PC-HBM token outputs back to the 28 x 28 p3 grid."""

from __future__ import annotations

from typing import Mapping

import torch

from ..common.utils import scatter_tokens


def pc_scatter(
    batch_size: int,
    height: int,
    width: int,
    batch_ids: torch.Tensor,
    flat_indices: torch.Tensor,
    token_aux: Mapping[str, torch.Tensor],
    gate_pc_token: torch.Tensor,
    c23_token: torch.Tensor,
    query_valid: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Scatter sparse tensors while dropping every invalid query row."""

    query_count = batch_ids.numel()
    if flat_indices.shape != batch_ids.shape:
        raise ValueError("batch_ids and flat_indices must have identical shape")
    if gate_pc_token.ndim == 1:
        gate_pc_token = gate_pc_token[:, None]
    if c23_token.ndim == 1:
        c23_token = c23_token[:, None]
    if gate_pc_token.shape != (query_count, 1) or c23_token.shape != (
        query_count,
        1,
    ):
        raise ValueError("gate_pc_token and c23_token must be [M,1]")
    if query_valid is None:
        valid_from_aux = token_aux.get("valid_token")
        if valid_from_aux is None:
            query_valid = torch.ones(
                query_count, device=batch_ids.device, dtype=torch.bool
            )
        else:
            query_valid = valid_from_aux.reshape(-1).to(
                device=batch_ids.device, dtype=torch.bool
            )
    else:
        query_valid = query_valid.to(device=batch_ids.device, dtype=torch.bool)
    if query_valid.shape != (query_count,):
        raise ValueError("query_valid must be [M]")
    keep = torch.nonzero(query_valid, as_tuple=False).flatten()
    kept_batch = batch_ids.index_select(0, keep)
    kept_indices = flat_indices.index_select(0, keep)

    def scatter(name: str, channels: int) -> torch.Tensor:
        values = token_aux[name].index_select(0, keep)
        return scatter_tokens(
            (batch_size, channels, height, width),
            kept_batch,
            kept_indices,
            values,
            reduce="replace",
        )

    shape1 = (batch_size, 1, height, width)
    gate_values = gate_pc_token.index_select(0, keep)
    c23_values = c23_token.index_select(0, keep)
    valid_values = gate_values.new_ones((keep.numel(), 1))
    return {
        "M_pc_map": scatter("M_pc_token", 1),
        "O_pc_map": scatter("O_pc_token", 2),
        "gate_pc_map": scatter_tokens(
            shape1, kept_batch, kept_indices, gate_values, reduce="replace"
        ),
        "C23_map": scatter_tokens(
            shape1, kept_batch, kept_indices, c23_values, reduce="replace"
        ),
        "Z3_map": scatter("Z3_token", token_aux["Z3_token"].size(1)),
        "E_attn_map": scatter("E_attn", token_aux["E_attn"].size(1)),
        "G_attn_map": scatter("G_attn", token_aux["G_attn"].size(1)),
        "valid3_map": scatter_tokens(
            shape1, kept_batch, kept_indices, valid_values, reduce="replace"
        ),
    }

