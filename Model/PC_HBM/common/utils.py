"""Batch-safe tensor helpers shared by DINO PC-HBM modules."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F


EPS = 1.0e-6
REGION_NAMES = ("fg_core", "fg_boundary", "bg_near", "bg_far")
REGION_TO_ID = {name: index for index, name in enumerate(REGION_NAMES)}


def finite_or_zero(value: torch.Tensor) -> torch.Tensor:
    """Replace non-finite entries while preserving shape, dtype and device."""

    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def normalize(value: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    return F.normalize(finite_or_zero(value), dim=dim, eps=eps)


def normalize_prob(value: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    """Normalize non-negative values; an all-zero row stays all zero."""

    value = finite_or_zero(value).clamp_min(0.0)
    denominator = value.sum(dim=dim, keepdim=True)
    return torch.where(
        denominator > eps,
        value / denominator.clamp_min(eps),
        torch.zeros_like(value),
    )


def entropy_from_probs(probability: torch.Tensor, dim: int = -1, eps: float = EPS) -> torch.Tensor:
    probability = normalize_prob(probability, dim=dim, eps=eps)
    entropy = -(probability * probability.clamp_min(eps).log()).sum(dim=dim)
    cardinality = probability.size(dim)
    if cardinality <= 1:
        return torch.zeros_like(entropy)
    return (entropy / math.log(cardinality)).clamp(0.0, 1.0)


def js_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    dim: int = -1,
    eps: float = EPS,
) -> torch.Tensor:
    p = normalize_prob(p, dim=dim, eps=eps)
    q = normalize_prob(q, dim=dim, eps=eps)
    midpoint = 0.5 * (p + q)
    p_term = p * (p / midpoint.clamp_min(eps)).clamp_min(eps).log()
    q_term = q * (q / midpoint.clamp_min(eps)).clamp_min(eps).log()
    result = 0.5 * (p_term.sum(dim=dim) + q_term.sum(dim=dim))
    valid = (p.sum(dim=dim) > eps) & (q.sum(dim=dim) > eps)
    return torch.where(valid, finite_or_zero(result), torch.zeros_like(result))


def masked_softmax(
    logits: torch.Tensor,
    mask: torch.Tensor | None,
    dim: int = -1,
) -> torch.Tensor:
    """Softmax over valid candidates; fully invalid rows return exact zeros."""

    logits = finite_or_zero(logits)
    if mask is None:
        return torch.softmax(logits, dim=dim)
    mask = mask.to(device=logits.device, dtype=torch.bool)
    if mask.shape != logits.shape:
        mask = torch.broadcast_to(mask, logits.shape)
    masked = logits.masked_fill(~mask, -1.0e4)
    probability = torch.softmax(masked, dim=dim) * mask.to(dtype=logits.dtype)
    denominator = probability.sum(dim=dim, keepdim=True)
    return torch.where(
        denominator > 0,
        probability / denominator.clamp_min(EPS),
        torch.zeros_like(probability),
    )


def safe_topk(logits: torch.Tensor, k: int, dim: int = -1) -> tuple[torch.Tensor, torch.Tensor]:
    if logits.size(dim) == 0 or int(k) <= 0:
        shape = list(logits.shape)
        shape[dim] = 0
        return logits.new_empty(shape), torch.empty(shape, device=logits.device, dtype=torch.long)
    return torch.topk(logits, k=min(int(k), logits.size(dim)), dim=dim)


def gradient_strength(probability: torch.Tensor) -> torch.Tensor:
    dx = F.pad(probability[..., :, 1:] - probability[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(probability[..., 1:, :] - probability[..., :-1, :], (0, 0, 0, 1))
    return torch.sqrt(dx.square() + dy.square() + EPS)


def morph_boundary(probability: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    padding = int(kernel_size) // 2
    dilated = F.max_pool2d(probability, kernel_size, stride=1, padding=padding)
    eroded = -F.max_pool2d(-probability, kernel_size, stride=1, padding=padding)
    return (dilated - eroded).clamp(0.0, 1.0)


def boundary_features_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    uncertainty = 4.0 * probability * (1.0 - probability)
    gradient = gradient_strength(probability)
    entropy = -(
        probability * probability.clamp_min(EPS).log()
        + (1.0 - probability) * (1.0 - probability).clamp_min(EPS).log()
    ) / math.log(2.0)
    return torch.cat(
        [
            morph_boundary(probability),
            uncertainty.clamp(0.0, 1.0),
            gradient,
            entropy.clamp(0.0, 1.0),
            probability,
        ],
        dim=1,
    )


def token_indices_from_score(
    score: torch.Tensor,
    top_ratio: float = 0.25,
    threshold: float | None = None,
    min_tokens: int = 1,
    max_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select a bounded, variable number of tokens independently per image."""

    if score.ndim != 4 or score.size(1) != 1:
        raise ValueError(f"score must be [B,1,H,W], got {tuple(score.shape)}")
    batch_size, _, height, width = score.shape
    flattened = finite_or_zero(score).flatten(2).squeeze(1)
    total = height * width
    default_count = max(int(min_tokens), int(round(total * float(top_ratio))))
    if max_tokens is not None:
        default_count = min(default_count, int(max_tokens))
    default_count = min(default_count, total)
    batch_ids: list[torch.Tensor] = []
    flat_indices: list[torch.Tensor] = []
    token_scores: list[torch.Tensor] = []
    for batch_index in range(batch_size):
        row = flattened[batch_index]
        if threshold is None:
            keep = torch.topk(row, k=default_count).indices
        else:
            keep = torch.nonzero(row >= float(threshold), as_tuple=False).flatten()
            if keep.numel() < int(min_tokens):
                keep = torch.topk(row, k=default_count).indices
            elif max_tokens is not None and keep.numel() > int(max_tokens):
                order = torch.topk(row.index_select(0, keep), k=int(max_tokens)).indices
                keep = keep.index_select(0, order)
        batch_ids.append(torch.full_like(keep, batch_index, dtype=torch.long))
        flat_indices.append(keep.long())
        token_scores.append(row.index_select(0, keep))
    if not batch_ids:
        empty_long = torch.empty(0, device=score.device, dtype=torch.long)
        return empty_long, empty_long, score.new_empty(0)
    return torch.cat(batch_ids), torch.cat(flat_indices), torch.cat(token_scores)


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
    _validate_token_indices(feature_map, batch_ids, flat_indices)
    if batch_ids.numel() == 0:
        return feature_map.new_empty((0, feature_map.size(1)))
    flattened = feature_map.flatten(2).transpose(1, 2).contiguous()
    return flattened[batch_ids.long(), flat_indices.long()]


def gather_local_patches(
    feature_map: torch.Tensor,
    batch_ids: torch.Tensor,
    flat_indices: torch.Tensor,
    window: int = 5,
) -> torch.Tensor:
    """Gather patches with one ``F.unfold`` call per represented image.

    This deliberately does not unfold the entire physical batch at once, and
    never performs a Python crop per token.
    """

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
        selected = selected.reshape(-1, channels, kernel, kernel)
        output.index_copy_(0, positions, selected)
    return output


def geometry_map_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Build six-channel prediction geometry from a single-channel logit map."""

    if logits.ndim != 4 or logits.size(1) != 1:
        raise ValueError(f"logits must be [B,1,H,W], got {tuple(logits.shape)}")
    probability = torch.sigmoid(logits)
    dx = F.pad(probability[..., :, 1:] - probability[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(probability[..., 1:, :] - probability[..., :-1, :], (0, 0, 0, 1))
    gradient_norm = torch.sqrt(dx.square() + dy.square() + EPS)
    normal_x = dx / gradient_norm
    normal_y = dy / gradient_norm
    sdf_proxy = 2.0 * probability - 1.0
    offset_x = -sdf_proxy * normal_x
    offset_y = -sdf_proxy * normal_y
    uncertainty = 4.0 * probability * (1.0 - probability)
    maximum = gradient_norm.amax(dim=(-2, -1), keepdim=True)
    reliability = gradient_norm / (maximum + EPS) * (1.0 - uncertainty)
    return torch.cat(
        [
            sdf_proxy,
            normal_x,
            normal_y,
            offset_x,
            offset_y,
            reliability.clamp(0.0, 1.0),
        ],
        dim=1,
    )


def scatter_tokens(
    shape: Iterable[int],
    batch_ids: torch.Tensor,
    flat_indices: torch.Tensor,
    values: torch.Tensor,
    reduce: str = "replace",
) -> torch.Tensor:
    batch_size, channels, height, width = [int(value) for value in shape]
    if values.ndim != 2 or values.size(1) != channels:
        raise ValueError(f"values must be [M,{channels}], got {tuple(values.shape)}")
    output = values.new_zeros((batch_size, channels, height * width))
    if values.numel() == 0:
        return output.view(batch_size, channels, height, width)
    if values.size(0) != batch_ids.numel() or batch_ids.shape != flat_indices.shape:
        raise ValueError("values, batch_ids and flat_indices must have the same leading length")
    for batch_index in range(batch_size):
        keep = torch.nonzero(batch_ids == batch_index, as_tuple=False).flatten()
        if keep.numel() == 0:
            continue
        positions = flat_indices.index_select(0, keep).long()
        selected = values.index_select(0, keep).transpose(0, 1)
        if reduce == "add":
            output[batch_index].index_add_(1, positions, selected)
        elif reduce == "replace":
            output[batch_index].index_copy_(1, positions, selected)
        else:
            raise ValueError(f"Unsupported scatter reduction: {reduce}")
    return output.view(batch_size, channels, height, width)


def add_tokens_to_map(
    base: torch.Tensor,
    batch_ids: torch.Tensor,
    flat_indices: torch.Tensor,
    delta: torch.Tensor,
) -> torch.Tensor:
    if delta.numel() == 0:
        return base
    correction = scatter_tokens(base.shape, batch_ids, flat_indices, delta, reduce="add")
    return base + correction


def scale_flat_indices(
    flat_indices: torch.Tensor,
    src_hw: Tuple[int, int],
    dst_hw: Tuple[int, int],
) -> torch.Tensor:
    src_height, src_width = (int(src_hw[0]), int(src_hw[1]))
    dst_height, dst_width = (int(dst_hw[0]), int(dst_hw[1]))
    if min(src_height, src_width, dst_height, dst_width) <= 0:
        raise ValueError("Source and destination grids must be non-empty")
    y = torch.div(flat_indices.long(), src_width, rounding_mode="floor")
    x = flat_indices.long().remainder(src_width)
    mapped_y = (y.float() * dst_height / src_height).floor().long().clamp(0, dst_height - 1)
    mapped_x = (x.float() * dst_width / src_width).floor().long().clamp(0, dst_width - 1)
    return mapped_y * dst_width + mapped_x


def local_window_gather(
    ref_map: torch.Tensor,
    query_batch_ids: torch.Tensor,
    query_flat_indices: torch.Tensor,
    query_hw: Tuple[int, int],
    ref_hw: Tuple[int, int],
    window: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather a small local reference window and its in-bounds mask."""

    count = int(query_flat_indices.numel())
    channels = ref_map.size(1)
    kernel = int(window)
    if kernel <= 0 or kernel % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if count == 0:
        return (
            ref_map.new_empty((0, kernel * kernel, channels)),
            torch.empty((0, kernel * kernel), device=ref_map.device, dtype=torch.bool),
        )
    query_height, query_width = map(int, query_hw)
    ref_height, ref_width = map(int, ref_hw)
    query_y = torch.div(query_flat_indices.long(), query_width, rounding_mode="floor")
    query_x = query_flat_indices.long().remainder(query_width)
    center_y = (query_y.float() * ref_height / query_height).floor().long().clamp(0, ref_height - 1)
    center_x = (query_x.float() * ref_width / query_width).floor().long().clamp(0, ref_width - 1)
    flattened = ref_map.flatten(2).transpose(1, 2).contiguous()
    refs: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    radius = kernel // 2
    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            y = center_y + offset_y
            x = center_x + offset_x
            valid = (y >= 0) & (y < ref_height) & (x >= 0) & (x < ref_width)
            index = y.clamp(0, ref_height - 1) * ref_width + x.clamp(0, ref_width - 1)
            refs.append(flattened[query_batch_ids.long(), index.long()])
            masks.append(valid)
    return torch.stack(refs, dim=1), torch.stack(masks, dim=1)


def make_normalized_grid(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    ys = (torch.arange(height, device=device, dtype=dtype) + 0.5) * 2.0 / height - 1.0
    xs = (torch.arange(width, device=device, dtype=dtype) + 0.5) * 2.0 / width - 1.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack((xx, yy), dim=-1).unsqueeze(0)


def merge_parent_results(
    results: Sequence[Mapping[str, Any]],
    total_queries: int,
    *,
    template: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Scatter independently routed retrieval rows back to query order.

    Each item must carry ``output_positions``.  Padded indices/pointers remain
    ``-1``, scores remain ``-1e4`` and validity remains ``False``.
    """

    source = template if template is not None else (results[0] if results else None)
    if source is None:
        return {
            "top_parent_valid": torch.empty((int(total_queries), 0), dtype=torch.bool),
            "top_parent_scores": torch.empty((int(total_queries), 0)),
            "top_parent_meta": [[] for _ in range(int(total_queries))],
        }
    merged: dict[str, Any] = {}
    integer_fill_keys = {"top_child_ptrs", "top_parent_indices", "top_parent_region_ids"}
    score_fill_keys = {"top_parent_scores"}
    skip_keys = {"output_positions", "q3_map"}
    for key, value in source.items():
        if key in skip_keys:
            continue
        if isinstance(value, torch.Tensor) and value.ndim >= 1:
            shape = (int(total_queries), *value.shape[1:])
            if value.dtype == torch.bool:
                merged[key] = torch.zeros(shape, device=value.device, dtype=torch.bool)
            elif key in integer_fill_keys or not value.dtype.is_floating_point:
                merged[key] = torch.full(shape, -1, device=value.device, dtype=value.dtype)
            elif key in score_fill_keys:
                merged[key] = torch.full(shape, -1.0e4, device=value.device, dtype=value.dtype)
            else:
                merged[key] = torch.zeros(shape, device=value.device, dtype=value.dtype)
        elif key == "top_parent_meta":
            merged[key] = [[] for _ in range(int(total_queries))]
    for result in results:
        positions = result["output_positions"].long()
        for key, destination in merged.items():
            if key not in result:
                continue
            value = result[key]
            if isinstance(destination, torch.Tensor):
                destination.index_copy_(0, positions.to(destination.device), value)
            elif key == "top_parent_meta":
                for output_index, metadata in zip(positions.detach().cpu().tolist(), value):
                    destination[output_index] = metadata
    return merged

