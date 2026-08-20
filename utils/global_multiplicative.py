from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch

from utils.checkpoint_pc_hbm import normalize_sample_key


GLOBAL_MULTIPLICATIVE_FORMULA_VERSION = (
    "global_multiplicative_v1_replicate_hypot_eps_denominator_only"
)
GLOBAL_MULTIPLICATIVE_PROTOCOL_VERSION = (
    "global_multiplicative_topk_kmeans_seed_v1"
)


@dataclass(frozen=True)
class GlobalMultiplicativeSelectionResult:
    """Deterministic global Top-K selections derived from a fixed seed split."""

    splits: dict[int, list[str]]
    selection_order: tuple[str, ...]
    global_rank: dict[str, int]
    selection_rank: dict[int, dict[str, int]]


def compute_global_multiplicative_score(
    boundary_disagreement: Any,
    global_disagreement: Any,
) -> torch.Tensor:
    """Compute ``D_bd * (1 - D_all)`` from aligned float32 components.

    The source components are required to be one-dimensional float32 tensors in
    ``[0, 1]``.  Requiring their original dtype prevents a lower-precision cache
    from silently changing the stable global ranking.
    """

    boundary = _validated_float32_vector(
        boundary_disagreement,
        name="boundary_disagreement",
    )
    global_value = _validated_float32_vector(
        global_disagreement,
        name="global_disagreement",
    )
    if boundary.shape != global_value.shape:
        raise ValueError(
            "boundary_disagreement and global_disagreement must have the same shape."
        )

    score = boundary * (1.0 - global_value)
    if not bool(torch.isfinite(score).all()):
        raise RuntimeError(
            "global multiplicative score unexpectedly contains NaN or infinity."
        )
    if bool(((score < 0.0) | (score > 1.0)).any()):
        raise RuntimeError(
            "global multiplicative score unexpectedly falls outside [0, 1]."
        )
    return score.contiguous()


def build_global_multiplicative_splits(
    sample_keys: Sequence[str],
    scores: torch.Tensor,
    seed_keys: Sequence[str],
    target_counts: Sequence[int] = (41, 202, 404, 808),
) -> GlobalMultiplicativeSelectionResult:
    """Select exact nested splits by global ``(-V, sample_key)`` ranking."""

    keys = _normalize_unique_keys(sample_keys, name="sample_keys")
    score_tensor = _validated_float32_vector(scores, name="scores")
    if score_tensor.numel() != len(keys):
        raise ValueError(
            f"scores must contain {len(keys)} elements aligned to sample_keys."
        )
    seeds = _normalize_unique_keys(seed_keys, name="seed_keys")
    targets = _validated_targets(target_counts, sample_count=len(keys))

    key_to_index = {key: index for index, key in enumerate(keys)}
    missing_seeds = sorted(set(seeds) - set(key_to_index))
    if missing_seeds:
        raise ValueError(f"seed_keys are absent from the catalog: {missing_seeds[:3]!r}")
    if len(seeds) > targets[0]:
        raise ValueError("seed count exceeds the smallest target.")

    seed_set = set(seeds)
    ranked_candidates = sorted(
        (key for key in keys if key not in seed_set),
        key=lambda key: (-float(score_tensor[key_to_index[key]]), key),
    )
    global_rank = {
        key: rank for rank, key in enumerate(ranked_candidates, start=1)
    }

    sorted_seeds = sorted(seed_set)
    selected_candidates = ranked_candidates[: targets[-1] - len(sorted_seeds)]
    selection_order = tuple(sorted_seeds + selected_candidates)
    overall_rank = {
        key: rank for rank, key in enumerate(selection_order, start=1)
    }

    splits: dict[int, list[str]] = {}
    selection_rank: dict[int, dict[str, int]] = {}
    previous_split: set[str] | None = None
    for target in targets:
        additional_count = target - len(sorted_seeds)
        selected = seed_set | set(ranked_candidates[:additional_count])
        if len(selected) != target:
            raise RuntimeError(
                "Global multiplicative selection produced "
                f"{len(selected)} samples for target {target}."
            )
        if previous_split is not None and not previous_split.issubset(selected):
            raise RuntimeError("Nested split invariant was violated.")

        split = sorted(selected)
        splits[target] = split
        selection_rank[target] = {key: overall_rank[key] for key in split}
        previous_split = selected

    return GlobalMultiplicativeSelectionResult(
        splits=splits,
        selection_order=selection_order,
        global_rank=global_rank,
        selection_rank=selection_rank,
    )


def labeled_names_from_keys(sample_keys: Sequence[str]) -> list[str]:
    """Convert stable dataset keys to ordered bare stems for TXT artifacts."""

    keys = _normalize_unique_keys(sample_keys, name="sample_keys")
    names = [key.rsplit("/", 1)[-1] for key in keys]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"sample keys produce duplicate labeled stems: {duplicates[:3]!r}")
    return names


def _validated_float32_vector(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if value.dtype != torch.float32:
        raise TypeError(f"{name} must have dtype torch.float32, got {value.dtype}.")
    tensor = value.detach().cpu().contiguous()
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN or infinity.")
    if bool(((tensor < 0.0) | (tensor > 1.0)).any()):
        raise ValueError(f"{name} must be in [0, 1].")
    return tensor


def _normalize_unique_keys(sample_keys: Sequence[str], *, name: str) -> list[str]:
    if isinstance(sample_keys, (str, bytes)) or not isinstance(sample_keys, Sequence):
        raise TypeError(f"{name} must be a sequence of strings.")
    keys: list[str] = []
    for value in sample_keys:
        if not isinstance(value, str):
            raise TypeError(f"{name} must contain only strings.")
        keys.append(normalize_sample_key(value))
    if len(keys) != len(set(keys)):
        raise ValueError(f"{name} contains duplicate keys.")
    return keys


def _validated_targets(
    target_counts: Sequence[int],
    *,
    sample_count: int,
) -> tuple[int, ...]:
    if isinstance(target_counts, (str, bytes)) or not isinstance(target_counts, Sequence):
        raise TypeError("target_counts must be a sequence of integers.")
    if not target_counts:
        raise ValueError("target_counts must not be empty.")
    if any(
        isinstance(target, bool) or not isinstance(target, Integral)
        for target in target_counts
    ):
        raise TypeError("target_counts must contain integers.")
    targets = tuple(int(target) for target in target_counts)
    if any(target <= 0 for target in targets):
        raise ValueError("target_counts must be positive.")
    if tuple(sorted(set(targets))) != targets:
        raise ValueError("target_counts must be strictly increasing.")
    if targets[-1] > sample_count:
        raise ValueError("largest target exceeds sample count.")
    return targets
