"""GT-derived PC-HBM supervision targets and token gather helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from ..memory.pc_region_builder import build_pc_regions


REGION_FG_CORE = 0
REGION_FG_BOUNDARY = 1
REGION_BG_NEAR = 2
REGION_BG_FAR = 3


def build_region_label_map(gt: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Return mutually-exclusive region labels ``[B,H,W]`` in ``0..3``."""

    regions = build_pc_regions(gt, target_size=size)
    labels = torch.full(
        (gt.size(0), int(size[0]), int(size[1])),
        REGION_BG_FAR,
        device=gt.device,
        dtype=torch.long,
    )
    labels[regions["bg_near"][:, 0] > 0.5] = REGION_BG_NEAR
    labels[regions["fg_core"][:, 0] > 0.5] = REGION_FG_CORE
    labels[regions["fg_boundary"][:, 0] > 0.5] = REGION_FG_BOUNDARY
    return labels


def build_geometry_target(gt: torch.Tensor, size: tuple[int, int]) -> dict[str, torch.Tensor]:
    """Return SDF, normal, offset and reliability maps at ``size``."""

    geometry = build_pc_regions(gt, target_size=size)["geometry"]
    return {
        "sdf": geometry[:, 0:1],
        "normal": geometry[:, 1:3],
        "offset": geometry[:, 3:5],
        "reliability": geometry[:, 5:6],
    }


def normalize_boundary_indices(
    indices: Any = None,
    *,
    batch_ids: torch.Tensor | None = None,
    flat_indices: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor] | None:
    """Normalize supported boundary-index representations.

    The DINO engine uses a mapping with ``batch_ids`` and ``flat_indices``;
    tests and older leaf modules may instead expose ``[M,2]`` or a two-tuple.
    """

    if isinstance(indices, Mapping):
        batch_ids = indices.get("batch_ids", indices.get("batch_ids3", batch_ids))
        flat_indices = indices.get("flat_indices", indices.get("flat_indices3", flat_indices))
    elif torch.is_tensor(indices):
        if indices.ndim != 2 or indices.size(-1) != 2:
            raise ValueError(f"boundary indices tensor must be [M,2], got {tuple(indices.shape)}")
        batch_ids, flat_indices = indices[:, 0], indices[:, 1]
    elif isinstance(indices, (tuple, list)) and len(indices) == 2:
        batch_ids, flat_indices = indices
    elif indices is not None:
        raise TypeError(f"Unsupported boundary indices: {type(indices).__name__}")

    if batch_ids is None or flat_indices is None:
        return None
    batch_ids = torch.as_tensor(batch_ids, device=device, dtype=torch.long).flatten()
    flat_indices = torch.as_tensor(flat_indices, device=device, dtype=torch.long).flatten()
    if batch_ids.numel() != flat_indices.numel():
        raise ValueError("batch_ids and flat_indices must have equal length")
    return {"batch_ids": batch_ids, "flat_indices": flat_indices}


def gather_by_boundary_indices(
    map_tensor: torch.Tensor,
    boundary_indices: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Gather ``[B,C,H,W]`` or ``[B,H,W]`` at flattened token indices."""

    normalized = normalize_boundary_indices(boundary_indices, device=map_tensor.device)
    if normalized is None:
        raise ValueError("boundary_indices are required")
    batch_ids = normalized["batch_ids"]
    flat_indices = normalized["flat_indices"]
    spatial_size = int(map_tensor.shape[-2] * map_tensor.shape[-1])
    if batch_ids.numel() and (
        int(batch_ids.min()) < 0
        or int(batch_ids.max()) >= map_tensor.size(0)
        or int(flat_indices.min()) < 0
        or int(flat_indices.max()) >= spatial_size
    ):
        raise IndexError("boundary token indices are outside the target map")
    if map_tensor.ndim == 4:
        channels = int(map_tensor.size(1))
        if not batch_ids.numel():
            return map_tensor.new_empty((0, channels))
        return map_tensor.flatten(2).transpose(1, 2)[batch_ids, flat_indices]
    if map_tensor.ndim == 3:
        if not batch_ids.numel():
            return map_tensor.new_empty((0,))
        return map_tensor.flatten(1)[batch_ids, flat_indices]
    raise ValueError(f"map_tensor must be [B,C,H,W] or [B,H,W], got {tuple(map_tensor.shape)}")


def build_need_correction_map(
    z_main: torch.Tensor,
    gt: torch.Tensor,
    size: tuple[int, int],
    threshold: float = 0.25,
) -> torch.Tensor:
    """Return pixels where detached main probability differs from GT."""

    if gt.ndim == 3:
        gt = gt.unsqueeze(1)
    p_main = torch.sigmoid(
        F.interpolate(z_main.detach(), size=size, mode="bilinear", align_corners=False)
    )
    gt_small = F.interpolate(gt.float(), size=size, mode="nearest")
    return ((p_main - gt_small).abs() > float(threshold)).to(dtype=z_main.dtype)


def build_gt_boundary(gt: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Return a three-pixel morphological GT boundary at ``size``."""

    if gt.ndim == 3:
        gt = gt.unsqueeze(1)
    target = F.interpolate(gt.float(), size=size, mode="nearest")
    dilation = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)
    erosion = 1.0 - F.max_pool2d(1.0 - target, kernel_size=3, stride=1, padding=1)
    return (dilation - erosion).clamp(0.0, 1.0)


__all__ = [
    "REGION_BG_FAR",
    "REGION_BG_NEAR",
    "REGION_FG_BOUNDARY",
    "REGION_FG_CORE",
    "build_geometry_target",
    "build_gt_boundary",
    "build_need_correction_map",
    "build_region_label_map",
    "gather_by_boundary_indices",
    "normalize_boundary_indices",
]
