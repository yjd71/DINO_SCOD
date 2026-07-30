"""Pure-PyTorch construction of the two PC-HBM-Lite memory regions."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


REGION_FG_BOUNDARY = 0
REGION_BG_NEAR = 1
REGION_IGNORE = -1
DEFAULT_REGION_NAMES = ("fg_boundary", "bg_near")
_RESERVED_OUTPUT_NAMES = {"pair_union", "pair_labels"}


def build_boundary_pair_regions(
    gt: torch.Tensor,
    target_size: tuple[int, int] = (28, 28),
    *,
    boundary_kernel: int = 3,
    bg_near_kernel: int = 7,
    threshold: float = 0.5,
    region_names: Sequence[str] = DEFAULT_REGION_NAMES,
) -> dict[str, torch.Tensor]:
    """Build foreground-boundary and near-background masks at token scale."""

    if gt.ndim == 3:
        gt = gt.unsqueeze(1)
    if gt.ndim != 4 or gt.size(1) != 1:
        raise ValueError(f"gt must be [B,1,H,W] or [B,H,W], got {tuple(gt.shape)}")
    height, width = (int(target_size[0]), int(target_size[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    for name, kernel in (
        ("boundary_kernel", int(boundary_kernel)),
        ("bg_near_kernel", int(bg_near_kernel)),
    ):
        if kernel <= 0 or kernel % 2 == 0:
            raise ValueError(f"{name} must be a positive odd integer")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0,1]")
    names = tuple(str(name) for name in region_names)
    if (
        len(names) != 2
        or any(not name for name in names)
        or len(set(names)) != 2
        or any(name in _RESERVED_OUTPUT_NAMES for name in names)
    ):
        raise ValueError(
            "region_names must contain two unique, non-empty, non-reserved names"
        )

    if gt.size(0) == 0:
        foreground = torch.empty(
            (0, 1, height, width),
            device=gt.device,
            dtype=torch.bool,
        )
    else:
        resized = F.interpolate(
            gt.detach().float(),
            size=(height, width),
            mode="nearest",
        )
        foreground = resized >= float(threshold)
    foreground_float = foreground.float()

    boundary_padding = int(boundary_kernel) // 2
    padded_background = F.pad(
        1.0 - foreground_float,
        (boundary_padding,) * 4,
        mode="constant",
        value=1.0,
    )
    eroded = 1.0 - F.max_pool2d(
        padded_background,
        kernel_size=int(boundary_kernel),
        stride=1,
    )
    fg_boundary = foreground & eroded.lt(0.5)

    background = ~foreground
    dilated = F.max_pool2d(
        foreground_float,
        kernel_size=int(bg_near_kernel),
        stride=1,
        padding=int(bg_near_kernel) // 2,
    ).gt(0.5)
    bg_near = background & dilated

    if bool((fg_boundary & bg_near).any()):
        raise RuntimeError("PC-HBM-Lite pair regions must not overlap")
    pair_union = fg_boundary | bg_near
    pair_labels = torch.full(
        (gt.size(0), height, width),
        REGION_IGNORE,
        device=gt.device,
        dtype=torch.long,
    )
    pair_labels[fg_boundary[:, 0]] = REGION_FG_BOUNDARY
    pair_labels[bg_near[:, 0]] = REGION_BG_NEAR
    return {
        names[REGION_FG_BOUNDARY]: fg_boundary,
        names[REGION_BG_NEAR]: bg_near,
        "pair_union": pair_union,
        "pair_labels": pair_labels,
    }


__all__ = [
    "REGION_BG_NEAR",
    "DEFAULT_REGION_NAMES",
    "REGION_FG_BOUNDARY",
    "REGION_IGNORE",
    "build_boundary_pair_regions",
]
