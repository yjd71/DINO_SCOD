"""Boundary-local flip disagreement and joint P2 prototype extraction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .prototype import build_boundary_prototype


BOUNDARY_SCORE_VERSION = "pc_bpus_boundary_v1_relu_gap_replicate_hypot"
SCORE_FORMULA_VERSION = BOUNDARY_SCORE_VERSION


@dataclass(frozen=True)
class BoundaryUtility:
    """Per-image components of the boundary-local utility value."""

    d_boundary: torch.Tensor
    d_global: torch.Tensor
    value: torch.Tensor
    boundary_mass: torch.Tensor
    valid: torch.Tensor
    boundary_weight: torch.Tensor
    disagreement: torch.Tensor


@dataclass(frozen=True)
class ScorePrototypeResult:
    """Batch result returned by :func:`score_and_prototype`."""

    d_boundary: torch.Tensor
    d_global: torch.Tensor
    value: torch.Tensor
    boundary_mass: torch.Tensor
    valid: torch.Tensor
    prototype: torch.Tensor

    @property
    def valid_boundary(self) -> torch.Tensor:
        """Alias used by cache and acquisition call sites."""

        return self.valid


def _validate_probability(probability: torch.Tensor, name: str) -> None:
    if not isinstance(probability, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if probability.ndim != 4 or probability.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B,1,H,W].")
    if probability.shape[0] <= 0 or probability.shape[-2] <= 0 or probability.shape[-1] <= 0:
        raise ValueError(f"{name} must have non-empty batch and spatial dimensions.")


def _validate_positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    return kernel_x, kernel_x.transpose(-1, -2).contiguous()


def sobel_magnitude(probability: torch.Tensor, *, eps: float = 1e-6) -> torch.Tensor:
    """Return a border-cleared Sobel magnitude using replicate padding.

    ``eps`` is validated for a consistent public API, but is deliberately not
    injected into the magnitude.  A constant prediction therefore has exactly
    zero boundary weight.
    """

    _validate_probability(probability, "probability")
    _validate_positive_finite(eps, "eps")
    probability = torch.nan_to_num(
        probability.float(), nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    kernel_x, kernel_y = _sobel_kernels(probability.device, probability.dtype)
    padded = F.pad(probability, (1, 1, 1, 1), mode="replicate")
    magnitude = torch.hypot(
        F.conv2d(padded, kernel_x),
        F.conv2d(padded, kernel_y),
    )

    # The outer response is excluded explicitly, including for non-constant
    # predictions, so padding cannot create a selectable pseudo-boundary.
    magnitude[..., 0, :] = 0.0
    magnitude[..., -1, :] = 0.0
    magnitude[..., :, 0] = 0.0
    magnitude[..., :, -1] = 0.0
    return magnitude


def compute_boundary_utility(
    probability: torch.Tensor,
    transformed_probability: torch.Tensor,
    *,
    eps: float = 1e-6,
    boundary_mass_eps: float = 1e-6,
) -> BoundaryUtility:
    """Compute boundary advantage ``relu(D_bd-D_all)*(1-D_all)``."""

    _validate_probability(probability, "probability")
    _validate_probability(transformed_probability, "transformed_probability")
    if probability.shape != transformed_probability.shape:
        raise ValueError("The two aligned probability views must have identical shapes.")
    eps = _validate_positive_finite(eps, "eps")
    boundary_mass_eps = _validate_positive_finite(
        boundary_mass_eps, "boundary_mass_eps"
    )

    probability = torch.nan_to_num(
        probability.float(), nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    transformed_probability = torch.nan_to_num(
        transformed_probability.float(), nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)

    mean_probability = 0.5 * (probability + transformed_probability)
    disagreement = (probability - transformed_probability).abs()
    boundary_weight = sobel_magnitude(mean_probability, eps=eps)
    boundary_mass = boundary_weight.flatten(1).sum(dim=1)
    boundary_numerator = (boundary_weight * disagreement).flatten(1).sum(dim=1)

    # Epsilon appears only in a denominator.  In particular, a zero Sobel
    # field retains exactly zero boundary disagreement and value.
    d_boundary = torch.nan_to_num(
        boundary_numerator / (boundary_mass + eps),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    d_global = torch.nan_to_num(
        disagreement.flatten(1).mean(dim=1),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    value = torch.nan_to_num(
        torch.relu(d_boundary - d_global) * (1.0 - d_global),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    valid = boundary_mass > boundary_mass_eps

    return BoundaryUtility(
        d_boundary=d_boundary,
        d_global=d_global,
        value=value,
        boundary_mass=boundary_mass,
        valid=valid,
        boundary_weight=boundary_weight,
        disagreement=disagreement,
    )


def _extract_pc_bpus_tensors(forward_result: Any, expected_batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(forward_result, (tuple, list)) or len(forward_result) != 2:
        raise ValueError("Selector must return (outputs, aux) when return_aux=True.")
    aux = forward_result[1]
    if not isinstance(aux, Mapping):
        raise ValueError("Selector auxiliary output must be a mapping.")
    probability = aux.get("p_final")
    features = aux.get("features")
    p2 = features.get("p2") if isinstance(features, Mapping) else None
    if not isinstance(probability, torch.Tensor):
        raise ValueError("Selector aux['p_final'] must be a tensor.")
    if not isinstance(p2, torch.Tensor):
        raise ValueError("Selector aux['features']['p2'] must be a tensor.")
    _validate_probability(probability, "aux['p_final']")
    if probability.shape[0] != expected_batch:
        raise ValueError("Selector p_final has an invalid batch dimension.")
    if p2.ndim != 4 or p2.shape[0] != expected_batch:
        raise ValueError("Selector P2 must have shape [2B,C,H,W].")
    if p2.shape[1] <= 0 or p2.shape[-2] <= 0 or p2.shape[-1] <= 0:
        raise ValueError("Selector P2 must have non-empty channel and spatial dimensions.")
    return probability, p2


@torch.no_grad()
def score_and_prototype(
    model,
    images: torch.Tensor,
    *,
    eps: float = 1e-6,
    boundary_mass_eps: float = 1e-6,
    use_amp: bool = False,
    expected_p2_shape: tuple[int, int, int] | None = None,
) -> ScorePrototypeResult:
    """Run aligned original/flip views and return utility plus a P2 prototype."""

    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("images must be a [B,C,H,W] tensor.")
    if images.shape[0] <= 0 or images.shape[1] <= 0:
        raise ValueError("images must have non-empty batch and channel dimensions.")
    _validate_positive_finite(eps, "eps")
    _validate_positive_finite(boundary_mass_eps, "boundary_mass_eps")

    batch_size = int(images.shape[0])
    two_views = torch.cat((images, torch.flip(images, dims=(-1,))), dim=0)
    if hasattr(model, "eval"):
        model.eval()
    amp_enabled = bool(use_amp and images.device.type == "cuda")
    with torch.autocast(device_type=images.device.type, enabled=amp_enabled):
        forward_result = model(two_views, pc_mode="off", return_aux=True)
    probabilities, p2 = _extract_pc_bpus_tensors(
        forward_result, expected_batch=2 * batch_size
    )
    if expected_p2_shape is not None:
        expected = tuple(int(value) for value in expected_p2_shape)
        if len(expected) != 3 or any(value <= 0 for value in expected):
            raise ValueError("expected_p2_shape must contain three positive integers.")
        if tuple(p2.shape[1:]) != expected:
            raise ValueError(
                f"Selector P2 must have shape [2B,{expected[0]},{expected[1]},{expected[2]}], "
                f"found {tuple(p2.shape)}."
            )

    probability, flipped_probability = probabilities.chunk(2, dim=0)
    transformed_probability = torch.flip(flipped_probability, dims=(-1,))
    p2_original, p2_flipped = p2.chunk(2, dim=0)
    p2_mean = 0.5 * (p2_original.float() + torch.flip(p2_flipped.float(), dims=(-1,)))

    utility = compute_boundary_utility(
        probability,
        transformed_probability,
        eps=eps,
        boundary_mass_eps=boundary_mass_eps,
    )
    prototype = build_boundary_prototype(
        p2_mean,
        utility.boundary_weight,
        valid_boundary=utility.valid,
        eps=eps,
        boundary_mass_eps=boundary_mass_eps,
    )
    return ScorePrototypeResult(
        d_boundary=utility.d_boundary,
        d_global=utility.d_global,
        value=utility.value,
        boundary_mass=utility.boundary_mass,
        valid=utility.valid,
        prototype=prototype,
    )
