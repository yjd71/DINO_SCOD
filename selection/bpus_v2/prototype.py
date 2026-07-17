"""Boundary-weighted P2 prototype construction for BPUS-v2."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


BPUS_V2_PROTOTYPE_VERSION = (
    "bpus_v2_p2_v2_aligned_mean_point_l2_bilinear_boundary_pool"
)
PROTOTYPE_LEVEL = "p2"
PROTOTYPE_DIM = 128
PROTOTYPE_HEIGHT = 28
PROTOTYPE_WIDTH = 28
EXPECTED_P2_SHAPE = (PROTOTYPE_DIM, PROTOTYPE_HEIGHT, PROTOTYPE_WIDTH)


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def build_bpus_v2_prototype(
    p2: torch.Tensor,
    boundary_weight: torch.Tensor,
    *,
    valid_boundary: torch.Tensor | None = None,
    eps: float = 1e-6,
    boundary_mass_eps: float = 1e-6,
) -> torch.Tensor:
    """Build exact ``[B,128]`` normalized boundary prototypes from P2.

    P2 features are normalized independently at every spatial location.  The
    predicted boundary map is resized with bilinear interpolation before the
    weighted aggregate is normalized once more.  Invalid rows remain exact
    zeros.
    """

    eps = _positive_finite(eps, "eps")
    boundary_mass_eps = _positive_finite(boundary_mass_eps, "boundary_mass_eps")
    if not isinstance(p2, torch.Tensor) or p2.ndim != 4:
        raise ValueError("p2 must have shape [B,128,28,28].")
    if tuple(p2.shape[1:]) != EXPECTED_P2_SHAPE or p2.shape[0] <= 0:
        raise ValueError(
            f"p2 must have shape [B,{PROTOTYPE_DIM},{PROTOTYPE_HEIGHT},"
            f"{PROTOTYPE_WIDTH}], found {tuple(p2.shape)}."
        )
    if not isinstance(boundary_weight, torch.Tensor):
        raise TypeError("boundary_weight must be a torch.Tensor.")
    if boundary_weight.ndim != 4 or boundary_weight.shape[1] != 1:
        raise ValueError("boundary_weight must have shape [B,1,H,W].")
    if boundary_weight.shape[0] != p2.shape[0]:
        raise ValueError("p2 and boundary_weight batch dimensions must match.")
    if boundary_weight.shape[-2] <= 0 or boundary_weight.shape[-1] <= 0:
        raise ValueError("boundary_weight must have non-empty spatial dimensions.")

    features = torch.nan_to_num(
        p2.float(), nan=0.0, posinf=0.0, neginf=0.0
    )
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

    if tuple(weights.shape[-2:]) != (PROTOTYPE_HEIGHT, PROTOTYPE_WIDTH):
        weights = F.interpolate(
            weights,
            size=(PROTOTYPE_HEIGHT, PROTOTYPE_WIDTH),
            mode="bilinear",
            align_corners=False,
        )

    point_norm = torch.linalg.vector_norm(features, dim=1, keepdim=True)
    unit_features = features / point_norm.clamp_min(eps)
    resized_mass = weights.flatten(1).sum(dim=1)
    pooled = (unit_features * weights).flatten(2).sum(dim=2)
    pooled = pooled / (resized_mass[:, None] + eps)
    pooled_norm = torch.linalg.vector_norm(pooled, dim=1, keepdim=True)
    prototype = pooled / pooled_norm.clamp_min(eps)
    prototype = torch.where(valid[:, None], prototype, torch.zeros_like(prototype))
    return torch.nan_to_num(prototype, nan=0.0, posinf=0.0, neginf=0.0)


PROTOTYPE_VERSION = BPUS_V2_PROTOTYPE_VERSION
build_boundary_prototype = build_bpus_v2_prototype
