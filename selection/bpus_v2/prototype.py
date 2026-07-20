"""Boundary-weighted P2 prototype construction for BPUS-v2."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


BPUS_V2_PROTOTYPE_VERSION = (
    "bpus_v2_p2_v3_strict_finite_aligned_mean_point_l2_bilinear_boundary_pool"
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


def _require_finite_tensor(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values (no NaN or Inf).")


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
    _require_finite_tensor(p2, "p2")
    if not isinstance(boundary_weight, torch.Tensor):
        raise TypeError("boundary_weight must be a torch.Tensor.")
    if boundary_weight.ndim != 4 or boundary_weight.shape[1] != 1:
        raise ValueError("boundary_weight must have shape [B,1,H,W].")
    if boundary_weight.shape[0] != p2.shape[0]:
        raise ValueError("p2 and boundary_weight batch dimensions must match.")
    if boundary_weight.shape[-2] <= 0 or boundary_weight.shape[-1] <= 0:
        raise ValueError("boundary_weight must have non-empty spatial dimensions.")
    _require_finite_tensor(boundary_weight, "boundary_weight")

    features = p2.float()
    weights = boundary_weight.float().clamp_min(0.0)
    _require_finite_tensor(features, "p2 after float32 conversion")
    _require_finite_tensor(weights, "boundary_weight after float32 conversion")
    original_mass = weights.flatten(1).sum(dim=1)
    _require_finite_tensor(original_mass, "boundary_weight mass")
    if valid_boundary is None:
        valid = original_mass > boundary_mass_eps
    else:
        if not isinstance(valid_boundary, torch.Tensor):
            raise TypeError("valid_boundary must be a torch.Tensor.")
        if valid_boundary.ndim != 1 or valid_boundary.shape[0] != p2.shape[0]:
            raise ValueError("valid_boundary must have shape [B].")
        if valid_boundary.is_complex():
            raise ValueError("valid_boundary must be real-valued.")
        _require_finite_tensor(valid_boundary, "valid_boundary")
        valid = valid_boundary.to(device=p2.device, dtype=torch.bool)

    if tuple(weights.shape[-2:]) != (PROTOTYPE_HEIGHT, PROTOTYPE_WIDTH):
        weights = F.interpolate(
            weights,
            size=(PROTOTYPE_HEIGHT, PROTOTYPE_WIDTH),
            mode="bilinear",
            align_corners=False,
        )
        _require_finite_tensor(weights, "resized boundary_weight")

    point_norm = torch.linalg.vector_norm(features, dim=1, keepdim=True)
    _require_finite_tensor(point_norm, "P2 point norm")
    unit_features = features / point_norm.clamp_min(eps)
    _require_finite_tensor(unit_features, "normalized P2")
    resized_mass = weights.flatten(1).sum(dim=1)
    _require_finite_tensor(resized_mass, "resized boundary_weight mass")
    weighted_features = unit_features * weights
    _require_finite_tensor(weighted_features, "boundary-weighted P2")
    pooled = weighted_features.flatten(2).sum(dim=2)
    _require_finite_tensor(pooled, "pooled P2 numerator")
    pooled = pooled / (resized_mass[:, None] + eps)
    _require_finite_tensor(pooled, "pooled P2")
    pooled_norm = torch.linalg.vector_norm(pooled, dim=1, keepdim=True)
    _require_finite_tensor(pooled_norm, "pooled P2 norm")
    prototype = pooled / pooled_norm.clamp_min(eps)
    _require_finite_tensor(prototype, "normalized prototype")
    prototype = torch.where(valid[:, None], prototype, torch.zeros_like(prototype))
    _require_finite_tensor(prototype, "prototype")
    return prototype


PROTOTYPE_VERSION = BPUS_V2_PROTOTYPE_VERSION
build_boundary_prototype = build_bpus_v2_prototype
