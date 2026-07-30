"""Two-region supervision helpers for PC-HBM-Lite."""

from __future__ import annotations

import torch

from Model.PC_HBM.memory.pc_region_builder import (
    build_boundary_pair_regions,
)


REGION_FG_BOUNDARY = 0
REGION_BG_NEAR = 1
REGION_IGNORE = -1


def build_pair_label_map(
    gt: torch.Tensor,
    size: tuple[int, int],
    *,
    boundary_kernel: int = 3,
    bg_near_kernel: int = 7,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Return ``[B,H,W]`` labels 0/1 and -1 outside both pair regions."""

    regions = build_boundary_pair_regions(
        gt,
        target_size=size,
        boundary_kernel=boundary_kernel,
        bg_near_kernel=bg_near_kernel,
        threshold=threshold,
    )
    labels = regions["pair_labels"]
    if labels.dtype != torch.long or labels.ndim != 3:
        raise RuntimeError("Pair region builder returned invalid labels")
    allowed = (
        (labels == REGION_FG_BOUNDARY)
        | (labels == REGION_BG_NEAR)
        | (labels == REGION_IGNORE)
    )
    if not bool(allowed.all()):
        raise RuntimeError("Pair label map contains an unsupported class")
    return labels


__all__ = [
    "REGION_BG_NEAR",
    "REGION_FG_BOUNDARY",
    "REGION_IGNORE",
    "build_pair_label_map",
]
