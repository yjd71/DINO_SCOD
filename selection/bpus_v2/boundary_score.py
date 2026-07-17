"""Flip disagreement scoring and joint P2 extraction for BPUS-v2."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .formulas import SMOOTH_VALUE, compute_boundary_value
from .prototype import EXPECTED_P2_SHAPE, build_bpus_v2_prototype


BPUS_V2_FORMULA_VERSION = (
    "bpus_v2_boundary_v2_smooth_value_soft_reward_sobel_div8_replicate_hypot"
)
SCORE_FORMULA_VERSION = BPUS_V2_FORMULA_VERSION


@dataclass(frozen=True)
class BoundaryScoreBPUSV2:
    """Per-image boundary and global disagreement components."""

    boundary_disagreement: torch.Tensor
    global_disagreement: torch.Tensor
    boundary_value: torch.Tensor
    boundary_mass: torch.Tensor
    valid_boundary: torch.Tensor
    boundary_weight: torch.Tensor
    disagreement: torch.Tensor


@dataclass(frozen=True)
class ScorePrototypeBPUSV2Result:
    """Batch output of :func:`score_and_prototype_bpus_v2`."""

    boundary_disagreement: torch.Tensor
    global_disagreement: torch.Tensor
    boundary_value: torch.Tensor
    boundary_mass: torch.Tensor
    valid_boundary: torch.Tensor
    prototype: torch.Tensor

    @property
    def d_boundary(self) -> torch.Tensor:
        return self.boundary_disagreement

    @property
    def d_global(self) -> torch.Tensor:
        return self.global_disagreement

    @property
    def value(self) -> torch.Tensor:
        return self.boundary_value

    @property
    def valid(self) -> torch.Tensor:
        return self.valid_boundary


def _validate_probability(probability: torch.Tensor, name: str) -> None:
    if not isinstance(probability, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if probability.ndim != 4 or probability.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B,1,H,W].")
    if probability.shape[0] <= 0 or probability.shape[-2] <= 0 or probability.shape[-1] <= 0:
        raise ValueError(f"{name} must have non-empty batch and spatial dimensions.")


def _positive_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def sobel_magnitude_bpus_v2(
    probability: torch.Tensor, *, eps: float = 1e-6
) -> torch.Tensor:
    """Return a replicate-padded, border-cleared Sobel magnitude.

    ``eps`` is only validated here.  It is deliberately absent from the
    magnitude so a spatially constant prediction retains an exact zero edge
    field.
    """

    _validate_probability(probability, "probability")
    _positive_finite(eps, "eps")
    probability = torch.nan_to_num(
        probability.float(), nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    padded = F.pad(probability, (1, 1, 1, 1), mode="replicate")
    # Group the positive and negative halves symmetrically before subtraction.
    # Besides being exactly the conventional /8 Sobel operator, this ordering
    # makes both halves bit-identical for a constant field and therefore keeps
    # its mathematical zero exact in float32.
    left = (
        padded[..., :-2, :-2]
        + 2.0 * padded[..., 1:-1, :-2]
        + padded[..., 2:, :-2]
    )
    right = (
        padded[..., :-2, 2:]
        + 2.0 * padded[..., 1:-1, 2:]
        + padded[..., 2:, 2:]
    )
    top = (
        padded[..., :-2, :-2]
        + 2.0 * padded[..., :-2, 1:-1]
        + padded[..., :-2, 2:]
    )
    bottom = (
        padded[..., 2:, :-2]
        + 2.0 * padded[..., 2:, 1:-1]
        + padded[..., 2:, 2:]
    )
    magnitude = torch.hypot((right - left) / 8.0, (bottom - top) / 8.0)
    magnitude[..., 0, :] = 0.0
    magnitude[..., -1, :] = 0.0
    magnitude[..., :, 0] = 0.0
    magnitude[..., :, -1] = 0.0
    return magnitude


def compute_boundary_score_bpus_v2(
    probability: torch.Tensor,
    transformed_probability: torch.Tensor,
    *,
    eps: float = 1e-6,
    boundary_mass_eps: float = 1e-6,
    value_mode: str = SMOOTH_VALUE,
) -> BoundaryScoreBPUSV2:
    """Compute aligned flip disagreement and a masked boundary value."""

    _validate_probability(probability, "probability")
    _validate_probability(transformed_probability, "transformed_probability")
    if probability.shape != transformed_probability.shape:
        raise ValueError("The two aligned probability views must have identical shapes.")
    eps = _positive_finite(eps, "eps")
    boundary_mass_eps = _positive_finite(boundary_mass_eps, "boundary_mass_eps")

    probability = torch.nan_to_num(
        probability.float(), nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    transformed_probability = torch.nan_to_num(
        transformed_probability.float(), nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    mean_probability = 0.5 * (probability + transformed_probability)
    disagreement = (probability - transformed_probability).abs()
    boundary_weight = sobel_magnitude_bpus_v2(mean_probability, eps=eps)
    boundary_mass = boundary_weight.flatten(1).sum(dim=1)
    numerator = (boundary_weight * disagreement).flatten(1).sum(dim=1)
    boundary_disagreement = torch.nan_to_num(
        numerator / (boundary_mass + eps), nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    global_disagreement = torch.nan_to_num(
        disagreement.flatten(1).mean(dim=1),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)
    valid_boundary = boundary_mass > boundary_mass_eps
    boundary_value = compute_boundary_value(
        boundary_disagreement,
        global_disagreement,
        valid_boundary,
        mode=value_mode,
    )
    return BoundaryScoreBPUSV2(
        boundary_disagreement=boundary_disagreement,
        global_disagreement=global_disagreement,
        boundary_value=boundary_value,
        boundary_mass=boundary_mass,
        valid_boundary=valid_boundary,
        boundary_weight=boundary_weight,
        disagreement=disagreement,
    )


def _extract_selector_tensors(
    forward_result: Any, expected_batch: int
) -> tuple[torch.Tensor, torch.Tensor]:
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
        raise ValueError("Selector P2 must have shape [B,128,28,28].")
    if tuple(p2.shape[1:]) != EXPECTED_P2_SHAPE:
        raise ValueError(
            f"Selector P2 must have shape [B,{EXPECTED_P2_SHAPE[0]},"
            f"{EXPECTED_P2_SHAPE[1]},{EXPECTED_P2_SHAPE[2]}], found {tuple(p2.shape)}."
        )
    return probability, p2


@torch.inference_mode()
def score_and_prototype_bpus_v2(
    model,
    images: torch.Tensor,
    *,
    eps: float = 1e-6,
    boundary_mass_eps: float = 1e-6,
    use_amp: bool = False,
    value_mode: str = SMOOTH_VALUE,
) -> ScorePrototypeBPUSV2Result:
    """Run two explicit aligned eval forwards and return value plus P2 prototype."""

    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("images must be a [B,C,H,W] tensor.")
    if images.shape[0] <= 0 or images.shape[1] <= 0:
        raise ValueError("images must have non-empty batch and channel dimensions.")
    _positive_finite(eps, "eps")
    _positive_finite(boundary_mass_eps, "boundary_mass_eps")
    batch_size = int(images.shape[0])
    if hasattr(model, "eval"):
        model.eval()
    amp_enabled = bool(use_amp and images.device.type == "cuda")

    with torch.autocast(device_type=images.device.type, enabled=amp_enabled):
        original_result = model(images, pc_mode="off", return_aux=True)
    with torch.autocast(device_type=images.device.type, enabled=amp_enabled):
        flipped_result = model(
            torch.flip(images, dims=(-1,)), pc_mode="off", return_aux=True
        )
    original_probability, original_p2 = _extract_selector_tensors(
        original_result, batch_size
    )
    flipped_probability, flipped_p2 = _extract_selector_tensors(
        flipped_result, batch_size
    )

    probability = original_probability.float()
    transformed_probability = torch.flip(flipped_probability.float(), dims=(-1,))
    aligned_flipped_p2 = torch.flip(flipped_p2.float(), dims=(-1,))
    mean_p2 = 0.5 * (original_p2.float() + aligned_flipped_p2)
    score = compute_boundary_score_bpus_v2(
        probability,
        transformed_probability,
        eps=eps,
        boundary_mass_eps=boundary_mass_eps,
        value_mode=value_mode,
    )
    prototype = build_bpus_v2_prototype(
        mean_p2,
        score.boundary_weight,
        valid_boundary=score.valid_boundary,
        eps=eps,
        boundary_mass_eps=boundary_mass_eps,
    )
    return ScorePrototypeBPUSV2Result(
        boundary_disagreement=score.boundary_disagreement,
        global_disagreement=score.global_disagreement,
        boundary_value=score.boundary_value,
        boundary_mass=score.boundary_mass,
        valid_boundary=score.valid_boundary,
        prototype=prototype,
    )


# Descriptive aliases make direct mathematical tests concise.
BOUNDARY_SCORE_VERSION = BPUS_V2_FORMULA_VERSION
sobel_magnitude = sobel_magnitude_bpus_v2
score_and_prototype = score_and_prototype_bpus_v2
