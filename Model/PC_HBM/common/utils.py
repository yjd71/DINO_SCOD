"""Small batch-safe tensor gathers shared by PC-HBM-Lite modules."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate_token_indices(
    feature_map: torch.Tensor,
    batch_ids: torch.Tensor,
    flat_indices: torch.Tensor,
) -> None:
    if feature_map.ndim != 4:
        raise ValueError(f"feature_map must be [B,C,H,W], got {tuple(feature_map.shape)}")
    if batch_ids.ndim != 1 or flat_indices.ndim != 1 or batch_ids.shape != flat_indices.shape:
        raise ValueError("batch_ids and flat_indices must be equal-length rank-1 tensors")
    if batch_ids.numel() == 0:
        return
    if int(batch_ids.min()) < 0 or int(batch_ids.max()) >= feature_map.size(0):
        raise IndexError("batch_ids contain an index outside feature_map")
    spatial_size = feature_map.size(2) * feature_map.size(3)
    if int(flat_indices.min()) < 0 or int(flat_indices.max()) >= spatial_size:
        raise IndexError("flat_indices contain an index outside feature_map")


def gather_tokens(
    feature_map: torch.Tensor,
    batch_ids: torch.Tensor,
    flat_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather aligned `[M,C]` tokens from a `[B,C,H,W]` feature map."""

    _validate_token_indices(feature_map, batch_ids, flat_indices)
    if batch_ids.numel() == 0:
        return feature_map.new_empty((0, feature_map.size(1)))
    flattened = feature_map.flatten(2).transpose(1, 2).contiguous()
    return flattened[batch_ids.long(), flat_indices.long()]


def gather_local_patches(
    feature_map: torch.Tensor,
    batch_ids: torch.Tensor,
    flat_indices: torch.Tensor,
    window: int = 3,
) -> torch.Tensor:
    """Gather aligned local patches without mixing physical batch elements."""

    _validate_token_indices(feature_map, batch_ids, flat_indices)
    kernel = int(window)
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError(f"window must be a positive odd integer, got {window}")
    count = int(batch_ids.numel())
    channels = feature_map.size(1)
    output = feature_map.new_empty((count, channels, kernel, kernel))
    if count == 0:
        return output

    for batch_index_tensor in batch_ids.unique(sorted=True):
        batch_index = int(batch_index_tensor)
        positions = torch.nonzero(batch_ids == batch_index, as_tuple=False).flatten()
        columns = F.unfold(
            feature_map[batch_index : batch_index + 1],
            kernel_size=kernel,
            padding=kernel // 2,
        ).squeeze(0).transpose(0, 1)
        selected = columns.index_select(0, flat_indices.index_select(0, positions).long())
        output.index_copy_(0, positions, selected.reshape(-1, channels, kernel, kernel))
    return output


__all__ = ["gather_local_patches", "gather_tokens"]
