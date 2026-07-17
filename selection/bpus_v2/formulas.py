"""Pure value, novelty, and utility formulas used by BPUS-v2."""

from __future__ import annotations

from dataclasses import dataclass

import torch


HARD_GAP = "hard-gap"
SMOOTH_VALUE = "smooth-value"
NOVELTY_GATE = "novelty-gate"
SOFT_REWARD = "soft-reward"


@dataclass(frozen=True)
class FormulaVariant:
    """A named pairing of a boundary value and novelty reward formula."""

    name: str
    value_mode: str
    reward_mode: str


FORMULA_VARIANTS: dict[str, FormulaVariant] = {
    "v1": FormulaVariant("v1", HARD_GAP, NOVELTY_GATE),
    "v2-a": FormulaVariant("v2-a", SMOOTH_VALUE, NOVELTY_GATE),
    "v2-b": FormulaVariant("v2-b", HARD_GAP, SOFT_REWARD),
    "v2": FormulaVariant("v2", SMOOTH_VALUE, SOFT_REWARD),
}


def resolve_formula_variant(name: str) -> FormulaVariant:
    """Resolve one of the four supported deterministic formula combinations."""

    try:
        return FORMULA_VARIANTS[str(name).lower()]
    except KeyError as error:
        choices = ", ".join(FORMULA_VARIANTS)
        raise ValueError(f"Unknown formula variant {name!r}; expected one of: {choices}.") from error


def _finite_float_tensor(value, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    tensor = value.float()
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite.")
    return tensor


def compute_boundary_value(
    boundary_disagreement: torch.Tensor,
    global_disagreement: torch.Tensor,
    valid_boundary: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    """Compute a masked boundary value for either supported value mode."""

    boundary = _finite_float_tensor(boundary_disagreement, "boundary_disagreement")
    global_value = _finite_float_tensor(global_disagreement, "global_disagreement")
    if not isinstance(valid_boundary, torch.Tensor):
        raise TypeError("valid_boundary must be a torch.Tensor.")
    try:
        boundary, global_value = torch.broadcast_tensors(boundary, global_value)
        valid = torch.broadcast_to(
            valid_boundary.to(device=boundary.device, dtype=torch.bool), boundary.shape
        )
    except RuntimeError as error:
        raise ValueError("Boundary inputs and valid_boundary must be broadcastable.") from error
    if bool(((boundary < 0.0) | (boundary > 1.0)).any()):
        raise ValueError("boundary_disagreement must lie in [0,1].")
    if bool(((global_value < 0.0) | (global_value > 1.0)).any()):
        raise ValueError("global_disagreement must lie in [0,1].")

    if mode == HARD_GAP:
        value = torch.relu(boundary - global_value) * (1.0 - global_value)
    elif mode == SMOOTH_VALUE:
        value = boundary * (1.0 - global_value)
    else:
        raise ValueError(f"Unknown value mode {mode!r}.")
    return torch.where(valid, value, torch.zeros_like(value)).clamp(0.0, 1.0)


def compute_bpus_v2_value(
    boundary_disagreement: torch.Tensor,
    global_disagreement: torch.Tensor,
    valid_boundary: torch.Tensor,
) -> torch.Tensor:
    """Compute the formal BPUS-v2 value ``D_bd * (1 - D_all)``."""

    return compute_boundary_value(
        boundary_disagreement,
        global_disagreement,
        valid_boundary,
        mode=SMOOTH_VALUE,
    )


def compute_bpus_v2_novelty(max_similarity: torch.Tensor) -> torch.Tensor:
    """Compute ``1 - clamp(max_similarity, 0, 1)`` in float32."""

    similarity = _finite_float_tensor(max_similarity, "max_similarity")
    return 1.0 - similarity.clamp(0.0, 1.0)


def compute_utility(
    value: torch.Tensor,
    novelty: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    """Combine value and novelty with a hard gate or a soft reward."""

    value_tensor = _finite_float_tensor(value, "value")
    novelty_tensor = _finite_float_tensor(novelty, "novelty")
    try:
        value_tensor, novelty_tensor = torch.broadcast_tensors(
            value_tensor, novelty_tensor
        )
    except RuntimeError as error:
        raise ValueError("value and novelty must be broadcastable.") from error
    if bool(((value_tensor < 0.0) | (value_tensor > 1.0)).any()):
        raise ValueError("value must lie in [0,1].")
    if bool(((novelty_tensor < 0.0) | (novelty_tensor > 1.0)).any()):
        raise ValueError("novelty must lie in [0,1].")

    if mode == NOVELTY_GATE:
        return value_tensor * novelty_tensor
    if mode == SOFT_REWARD:
        return value_tensor * (1.0 + novelty_tensor)
    raise ValueError(f"Unknown reward mode {mode!r}.")


def compute_bpus_v2_utility(
    value: torch.Tensor, novelty: torch.Tensor
) -> torch.Tensor:
    """Compute the formal BPUS-v2 utility ``V * (1 + N)``."""

    return compute_utility(value, novelty, mode=SOFT_REWARD)


def compute_variant_value(
    boundary_disagreement: torch.Tensor,
    global_disagreement: torch.Tensor,
    valid_boundary: torch.Tensor,
    *,
    variant: str,
) -> torch.Tensor:
    """Compute the value component selected by a named ablation variant."""

    spec = resolve_formula_variant(variant)
    return compute_boundary_value(
        boundary_disagreement,
        global_disagreement,
        valid_boundary,
        mode=spec.value_mode,
    )


def compute_variant_utility(
    value: torch.Tensor, novelty: torch.Tensor, *, variant: str
) -> torch.Tensor:
    """Compute the utility selected by a named ablation variant."""

    spec = resolve_formula_variant(variant)
    return compute_utility(value, novelty, mode=spec.reward_mode)
