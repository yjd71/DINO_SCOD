"""Boundary-weighted P2 prototype construction."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


PROTOTYPE_VERSION = "pc_bpus_p2_v1_aligned_mean_point_l2_boundary_pool"


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def build_boundary_prototype(
    p2: torch.Tensor,
    boundary_weight: torch.Tensor,
    *,
    valid_boundary: torch.Tensor | None = None,
    eps: float = 1e-6,
    boundary_mass_eps: float = 1e-6,
) -> torch.Tensor:
    """Pool point-normalized P2 features with the predicted boundary weights.

    The returned tensor is float32 ``[B,C]``.  Every valid non-zero aggregate
    is L2 normalized; every invalid-boundary row is exactly zero.
    """

    eps = _positive_finite(eps, "eps")
    boundary_mass_eps = _positive_finite(boundary_mass_eps, "boundary_mass_eps")
    if not isinstance(p2, torch.Tensor) or p2.ndim != 4:
        raise ValueError("p2 must have shape [B,C,H,W].")
    if p2.shape[0] <= 0 or p2.shape[1] <= 0 or p2.shape[-2] <= 0 or p2.shape[-1] <= 0:
        raise ValueError("p2 must have non-empty dimensions.")
    if not isinstance(boundary_weight, torch.Tensor):
        raise TypeError("boundary_weight must be a torch.Tensor.")
    if boundary_weight.ndim != 4 or boundary_weight.shape[1] != 1:
        raise ValueError("boundary_weight must have shape [B,1,H,W].")
    if boundary_weight.shape[0] != p2.shape[0]:
        raise ValueError("p2 and boundary_weight batch dimensions must match.")

    features = torch.nan_to_num(p2.float(), nan=0.0, posinf=0.0, neginf=0.0)
    weights = torch.nan_to_num(
        boundary_weight.float(), nan=0.0, posinf=0.0, neginf=0.0
    ).clamp_min(0.0)
    original_mass = weights.flatten(1).sum(dim=1)
    if valid_boundary is None:
        valid = original_mass > boundary_mass_eps
    else:
        if not isinstance(valid_boundary, torch.Tensor):
            raise TypeError("valid_boundary must be a torch.Tensor.")
        if valid_boundary.ndim != 1 or valid_boundary.shape[0] != p2.shape[0]:
            raise ValueError("valid_boundary must have shape [B].")
        valid = valid_boundary.to(device=p2.device, dtype=torch.bool)

    target_hw = (int(p2.shape[-2]), int(p2.shape[-1]))
    if tuple(weights.shape[-2:]) != target_hw:
        # Area resampling retains support from thin high-resolution edges and
        # only changes a common scale that cancels in weighted averaging.
        weights = F.interpolate(weights, size=target_hw, mode="area")

    point_norm = torch.linalg.vector_norm(features, dim=1, keepdim=True)
    unit_features = features / point_norm.clamp_min(eps)
    resized_mass = weights.flatten(1).sum(dim=1)
    pooled = (unit_features * weights).flatten(2).sum(dim=2)
    pooled = pooled / (resized_mass[:, None] + eps)
    pooled_norm = torch.linalg.vector_norm(pooled, dim=1, keepdim=True)
    prototype = pooled / pooled_norm.clamp_min(eps)
    prototype = torch.where(valid[:, None], prototype, torch.zeros_like(prototype))
    return torch.nan_to_num(prototype, nan=0.0, posinf=0.0, neginf=0.0)
