"""Build the four mutually-exclusive labelled PC-HBM regions and GT geometry."""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def build_pc_regions(
    gt: torch.Tensor,
    target_size: Tuple[int, int] = (28, 28),
    *,
    boundary_kernel: int = 3,
    bg_near_kernel: int = 7,
    threshold: float = 0.5,
    reliability_scale: float = 0.15,
) -> Dict[str, torch.Tensor]:
    """Resize GT with nearest-neighbour and construct regions/SDF geometry.

    Geometry channels are ``sdf, normal_x, normal_y, offset_x, offset_y,
    reliability``.  OpenCV distance transforms are evaluated on CPU, then the
    compact 28x28 result is returned to the input device.
    """

    if gt.ndim == 3:
        gt = gt.unsqueeze(1)
    if gt.ndim != 4 or gt.size(1) != 1:
        raise ValueError(f"gt must be [B,1,H,W] or [B,H,W], got {tuple(gt.shape)}")
    height, width = int(target_size[0]), int(target_size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    if boundary_kernel <= 0 or boundary_kernel % 2 == 0:
        raise ValueError("boundary_kernel must be a positive odd integer")
    if bg_near_kernel <= 0 or bg_near_kernel % 2 == 0:
        raise ValueError("bg_near_kernel must be a positive odd integer")
    if reliability_scale <= 0:
        raise ValueError("reliability_scale must be positive")

    device = gt.device
    gt_small = F.interpolate(gt.detach().float(), size=(height, width), mode="nearest")
    foreground = gt_small >= float(threshold)
    foreground_float = foreground.float()

    # A 3x3 *foreground* boundary is the foreground removed by erosion.
    eroded = 1.0 - F.max_pool2d(
        1.0 - foreground_float,
        kernel_size=boundary_kernel,
        stride=1,
        padding=boundary_kernel // 2,
    )
    fg_boundary = foreground & (eroded < 0.5)
    fg_core = foreground & ~fg_boundary

    background = ~foreground
    dilated = F.max_pool2d(
        foreground_float,
        kernel_size=bg_near_kernel,
        stride=1,
        padding=bg_near_kernel // 2,
    ) > 0.5
    bg_near = background & dilated
    bg_far = background & ~bg_near

    # Explicitly assert the partition invariant before deriving supervision.
    partition = torch.cat((fg_core, fg_boundary, bg_near, bg_far), dim=1)
    membership_count = partition.to(torch.int8).sum(dim=1, keepdim=True)
    if not bool(torch.all(membership_count == 1)):
        raise RuntimeError("PC-HBM regions must be mutually exclusive and cover every pixel")

    foreground_cpu = foreground.squeeze(1).to(device="cpu", dtype=torch.uint8).numpy()
    geometry_items = [_opencv_geometry(mask, reliability_scale) for mask in foreground_cpu]
    geometry = torch.from_numpy(np.stack(geometry_items, axis=0)).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    sdf = geometry[:, 0:1]
    return {
        "fg_core": fg_core.float(),
        "fg_boundary": fg_boundary.float(),
        "bg_near": bg_near.float(),
        "bg_far": bg_far.float(),
        "fg": foreground_float,
        "bg": background.float(),
        "boundary": fg_boundary.float(),
        "sdf": sdf,
        "geometry": geometry,
    }


def _opencv_geometry(foreground: np.ndarray, reliability_scale: float) -> np.ndarray:
    foreground = np.ascontiguousarray(foreground.astype(np.uint8, copy=False))
    height, width = foreground.shape
    if foreground.max() == 0:
        sdf = np.full((height, width), -1.0, dtype=np.float32)
    elif foreground.min() == 1:
        sdf = np.full((height, width), 1.0, dtype=np.float32)
    else:
        distance_to_background = cv2.distanceTransform(foreground, cv2.DIST_L2, 5)
        distance_to_foreground = cv2.distanceTransform(1 - foreground, cv2.DIST_L2, 5)
        sdf = (distance_to_background - distance_to_foreground) / float(max(height, width))
        sdf = np.clip(sdf, -1.0, 1.0).astype(np.float32, copy=False)

    # np.gradient returns dy then dx.  Flat degenerate masks correctly produce
    # a zero normal and therefore a zero offset.
    normal_y_raw, normal_x_raw = np.gradient(sdf)
    norm = np.sqrt(normal_x_raw * normal_x_raw + normal_y_raw * normal_y_raw + 1.0e-6)
    normal_x = normal_x_raw / norm
    normal_y = normal_y_raw / norm
    offset_x = -sdf * normal_x
    offset_y = -sdf * normal_y
    reliability = np.exp(-np.abs(sdf) / float(reliability_scale))
    return np.stack(
        (sdf, normal_x, normal_y, offset_x, offset_y, reliability),
        axis=0,
    ).astype(np.float32, copy=False)


__all__ = ["build_pc_regions"]

