from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import torch

from utils.checkpoint_pc_hbm import normalize_sample_key


GLOBAL_ADDITIVE_FORMULA_VERSION = "global_additive_v1_replicate_hypot"
GLOBAL_ADDITIVE_DEDUP_PROTOCOL_VERSION = "global_additive_global_dino_dedup_v1"


@dataclass(frozen=True)
class GlobalAdditiveSelectionResult:
    """Deterministic global Top-K selections derived from a fixed seed split."""

    splits: dict[int, list[str]]
    selection_order: tuple[str, ...]
    global_rank: dict[str, int]
    selection_rank: dict[int, dict[str, int]]


@dataclass(frozen=True)
class GlobalAdditiveDedupDecision:
    """Final audit state for one catalog key in the deduplicated stream."""

    decision: str
    global_rank: int | None
    max_cosine_similarity: float | None
    reference_key: str | None
    relaxed: bool
    evaluated_target_count: int | None
    selected_target_count: int | None


@dataclass(frozen=True)
class GlobalAdditiveDedupRound:
    """Per-target and cumulative counters for global DINO deduplication."""

    target_count: int
    budget: int
    strict_selected_count: int
    skipped_count: int
    relaxed_selected_count: int
    cumulative_strict_selected_count: int
    cumulative_skipped_count: int
    cumulative_relaxed_selected_count: int


@dataclass(frozen=True)
class GlobalAdditiveDedupSelectionResult:
    """Exact nested splits plus a complete per-key deduplication audit."""

    splits: dict[int, list[str]]
    selection_order: tuple[str, ...]
    global_rank: dict[str, int]
    selection_rank: dict[int, dict[str, int]]
    decisions: dict[str, GlobalAdditiveDedupDecision]
    rounds: tuple[GlobalAdditiveDedupRound, ...]
    dedup_threshold: float
    protocol_version: str = GLOBAL_ADDITIVE_DEDUP_PROTOCOL_VERSION

    @property
    def audit(self) -> dict[str, GlobalAdditiveDedupDecision]:
        """Compatibility alias exposing decisions as the per-key audit map."""

        return self.decisions


def compute_global_additive_score(
    boundary_disagreement: Any,
    global_disagreement: Any,
) -> torch.Tensor:
    """Compute ``D_bd + (1 - D_all)`` as a one-dimensional CPU float32 tensor.

    Both disagreement components are probabilities and therefore must be finite,
    aligned one-dimensional vectors in ``[0, 1]``.  The additive result remains
    in ``[0, 2]``; values above one are intentionally preserved.
    """

    boundary = _validated_component(
        boundary_disagreement,
        name="boundary_disagreement",
    )
    global_value = _validated_component(
        global_disagreement,
        name="global_disagreement",
    )
    if boundary.shape != global_value.shape:
        raise ValueError(
            "boundary_disagreement and global_disagreement must have the same shape."
        )

    score = boundary + (1.0 - global_value)
    if not bool(torch.isfinite(score).all()):
        raise RuntimeError("global additive score unexpectedly contains NaN or infinity.")
    if bool(((score < 0.0) | (score > 2.0)).any()):
        raise RuntimeError("global additive score unexpectedly falls outside [0, 2].")
    return score.contiguous()


def build_global_nested_splits(
    sample_keys: Sequence[str],
    scores: Sequence[float] | torch.Tensor,
    seed_keys: Sequence[str],
    target_counts: Sequence[int] = (41, 202, 404, 808),
) -> GlobalAdditiveSelectionResult:
    """Select exact nested splits by global ``(-score, sample_key)`` ranking."""

    keys = _normalize_unique_keys(sample_keys, name="sample_keys")
    score_tensor = _validated_scores(scores, expected_length=len(keys))
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
                f"Global additive selection produced {len(selected)} samples for target {target}."
            )
        if previous_split is not None and not previous_split.issubset(selected):
            raise RuntimeError("Nested split invariant was violated.")

        split = sorted(selected)
        splits[target] = split
        selection_rank[target] = {key: overall_rank[key] for key in split}
        previous_split = selected

    return GlobalAdditiveSelectionResult(
        splits=splits,
        selection_order=selection_order,
        global_rank=global_rank,
        selection_rank=selection_rank,
    )


def normalize_global_dino_features(
    features: Any,
    *,
    expected_length: int | None = None,
) -> torch.Tensor:
    """Validate and L2-normalize DINO rows as a CPU float32 matrix."""

    try:
        tensor = torch.as_tensor(features).detach().cpu()
    except (TypeError, ValueError, RuntimeError) as error:
        raise TypeError("features must be convertible to a numeric tensor.") from error
    if tensor.ndim != 2:
        raise ValueError("features must have shape [N, D].")
    if expected_length is not None and tensor.shape[0] != expected_length:
        raise ValueError(
            f"features must have {expected_length} rows aligned to sample_keys."
        )
    if tensor.shape[1] == 0:
        raise ValueError("features must have a non-empty feature dimension.")
    if torch.is_complex(tensor):
        raise TypeError("features must contain real values.")
    try:
        finite = torch.isfinite(tensor)
    except RuntimeError as error:
        raise TypeError("features must contain numeric values.") from error
    if not bool(finite.all()):
        raise ValueError("features contain NaN or infinity.")

    tensor = tensor.to(dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("features are not finite after float32 conversion.")
    norms = torch.linalg.vector_norm(tensor, ord=2, dim=1)
    zero_rows = torch.nonzero(norms == 0.0, as_tuple=False).flatten()
    if zero_rows.numel():
        preview = zero_rows[:3].tolist()
        raise ValueError(f"features contain zero-norm rows: {preview!r}")

    normalized = tensor / norms.unsqueeze(1)
    if not bool(torch.isfinite(normalized).all()):
        raise RuntimeError("normalized features unexpectedly contain NaN or infinity.")
    return normalized.contiguous()


def build_global_deduplicated_nested_splits(
    sample_keys: Sequence[str],
    scores: Sequence[float] | torch.Tensor,
    seed_keys: Sequence[str],
    features: Any,
    target_counts: Sequence[int] = (41, 202, 404, 808),
    dedup_threshold: float = 0.95,
) -> GlobalAdditiveDedupSelectionResult:
    """Build one nested global stream with strict DINO dedup and relaxed fill.

    Candidates are visited once in ``(-score, sample_key)`` order across every
    target.  A candidate is strictly accepted only when its maximum cosine to
    every seed and previously selected sample is not greater than the threshold.
    If the strict stream is exhausted, skipped candidates are backfilled in their
    original global rank so every requested target remains exact.
    """

    keys = _normalize_unique_keys(sample_keys, name="sample_keys")
    score_tensor = _validated_scores(scores, expected_length=len(keys))
    seeds = _normalize_unique_keys(seed_keys, name="seed_keys")
    if not seeds:
        raise ValueError("seed_keys must not be empty for DINO deduplication.")
    targets = _validated_targets(target_counts, sample_count=len(keys))
    threshold = _validated_dedup_threshold(dedup_threshold)
    normalized_features = normalize_global_dino_features(
        features,
        expected_length=len(keys),
    )

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

    decisions = {
        key: GlobalAdditiveDedupDecision(
            decision="seed" if key in seed_set else "not_evaluated",
            global_rank=global_rank.get(key),
            max_cosine_similarity=None,
            reference_key=None,
            relaxed=False,
            evaluated_target_count=None,
            selected_target_count=None,
        )
        for key in keys
    }
    selected = set(seed_set)
    selection_order = sorted(seed_set)
    skipped_candidates: list[str] = []
    cursor = 0
    cumulative_strict = 0
    cumulative_skipped = 0
    cumulative_relaxed = 0
    splits: dict[int, list[str]] = {}
    rounds: list[GlobalAdditiveDedupRound] = []
    previous_split: set[str] | None = None

    for target in targets:
        budget = target - len(selected)
        if budget < 0:
            raise RuntimeError("Current nested split already exceeds the next target.")
        round_strict = 0
        round_skipped = 0
        round_relaxed = 0

        while len(selected) < target and cursor < len(ranked_candidates):
            key = ranked_candidates[cursor]
            cursor += 1
            max_similarity, reference_key = _maximum_selected_similarity(
                key,
                selected_keys=selected,
                key_to_index=key_to_index,
                normalized_features=normalized_features,
            )
            if max_similarity > threshold:
                decisions[key] = GlobalAdditiveDedupDecision(
                    decision="duplicate_skipped",
                    global_rank=global_rank[key],
                    max_cosine_similarity=max_similarity,
                    reference_key=reference_key,
                    relaxed=False,
                    evaluated_target_count=target,
                    selected_target_count=None,
                )
                skipped_candidates.append(key)
                round_skipped += 1
                cumulative_skipped += 1
                continue

            selected.add(key)
            selection_order.append(key)
            decisions[key] = GlobalAdditiveDedupDecision(
                decision="strict_selected",
                global_rank=global_rank[key],
                max_cosine_similarity=max_similarity,
                reference_key=reference_key,
                relaxed=False,
                evaluated_target_count=target,
                selected_target_count=target,
            )
            round_strict += 1
            cumulative_strict += 1

        if len(selected) < target:
            for key in skipped_candidates:
                if key in selected:
                    continue
                previous_decision = decisions[key]
                selected.add(key)
                selection_order.append(key)
                decisions[key] = GlobalAdditiveDedupDecision(
                    decision="relaxed_backfill",
                    global_rank=previous_decision.global_rank,
                    max_cosine_similarity=previous_decision.max_cosine_similarity,
                    reference_key=previous_decision.reference_key,
                    relaxed=True,
                    evaluated_target_count=previous_decision.evaluated_target_count,
                    selected_target_count=target,
                )
                round_relaxed += 1
                cumulative_relaxed += 1
                if len(selected) == target:
                    break

        if len(selected) != target:
            raise RuntimeError(
                f"Global additive DINO dedup selected {len(selected)} samples "
                f"for target {target}."
            )
        current_split = set(selected)
        if previous_split is not None and not previous_split.issubset(current_split):
            raise RuntimeError("Nested split invariant was violated.")
        splits[target] = sorted(current_split)
        rounds.append(
            GlobalAdditiveDedupRound(
                target_count=target,
                budget=budget,
                strict_selected_count=round_strict,
                skipped_count=round_skipped,
                relaxed_selected_count=round_relaxed,
                cumulative_strict_selected_count=cumulative_strict,
                cumulative_skipped_count=cumulative_skipped,
                cumulative_relaxed_selected_count=cumulative_relaxed,
            )
        )
        previous_split = current_split

    overall_rank = {
        key: rank for rank, key in enumerate(selection_order, start=1)
    }
    return GlobalAdditiveDedupSelectionResult(
        splits=splits,
        selection_order=tuple(selection_order),
        global_rank=global_rank,
        selection_rank={
            target: {key: overall_rank[key] for key in split}
            for target, split in splits.items()
        },
        decisions=decisions,
        rounds=tuple(rounds),
        dedup_threshold=threshold,
    )


def _validated_dedup_threshold(value: Any) -> float:
    if isinstance(value, bool):
        raise TypeError("dedup_threshold must be a real number.")
    try:
        threshold = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError("dedup_threshold must be a real number.") from error
    if not math.isfinite(threshold):
        raise ValueError("dedup_threshold must be finite.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("dedup_threshold must be in [0, 1].")
    return threshold


def _maximum_selected_similarity(
    key: str,
    *,
    selected_keys: set[str],
    key_to_index: dict[str, int],
    normalized_features: torch.Tensor,
) -> tuple[float, str]:
    if not selected_keys:
        raise RuntimeError("DINO deduplication requires at least one selected reference.")
    reference_keys = sorted(selected_keys)
    reference_indices = [key_to_index[reference_key] for reference_key in reference_keys]
    similarities = normalized_features[reference_indices] @ normalized_features[key_to_index[key]]
    similarities = similarities.clamp(min=-1.0, max=1.0)
    reference_position = int(torch.argmax(similarities).item())
    return (
        float(similarities[reference_position].item()),
        reference_keys[reference_position],
    )


def labeled_names_from_keys(sample_keys: Sequence[str]) -> list[str]:
    """Convert stable dataset keys to ordered bare stems for TXT artifacts."""

    keys = _normalize_unique_keys(sample_keys, name="sample_keys")
    # Stable sample keys are already extension-free image stems.  Do not apply
    # Path.stem again: a legitimate source stem such as ``animal.v2`` must stay
    # ``animal.v2`` rather than being shortened to ``animal``.
    names = [key.rsplit("/", 1)[-1] for key in keys]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"sample keys produce duplicate labeled stems: {duplicates[:3]!r}")
    return names


def _validated_component(value: Any, *, name: str) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value).detach().cpu()
    except (TypeError, ValueError, RuntimeError) as error:
        raise TypeError(f"{name} must be convertible to a numeric tensor.") from error
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if torch.is_complex(tensor):
        raise TypeError(f"{name} must contain real values.")
    try:
        finite = torch.isfinite(tensor)
    except RuntimeError as error:
        raise TypeError(f"{name} must contain numeric values.") from error
    if not bool(finite.all()):
        raise ValueError(f"{name} contains NaN or infinity.")
    if bool(((tensor < 0.0) | (tensor > 1.0)).any()):
        raise ValueError(f"{name} must be in [0, 1].")
    return tensor.to(dtype=torch.float32).contiguous()


def _validated_scores(value: Any, *, expected_length: int) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu().contiguous()
    except (TypeError, ValueError, RuntimeError) as error:
        raise TypeError("scores must be convertible to a numeric tensor.") from error
    if tensor.ndim != 1 or tensor.numel() != expected_length:
        raise ValueError(
            f"scores must be a one-dimensional vector with {expected_length} elements."
        )
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("scores contain NaN or infinity.")
    if bool(((tensor < 0.0) | (tensor > 2.0)).any()):
        raise ValueError("scores must be in [0, 2].")
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
    if any(isinstance(target, bool) or not isinstance(target, Integral) for target in target_counts):
        raise TypeError("target_counts must contain integers.")
    targets = tuple(int(target) for target in target_counts)
    if any(target <= 0 for target in targets):
        raise ValueError("target_counts must be positive.")
    if tuple(sorted(set(targets))) != targets:
        raise ValueError("target_counts must be strictly increasing.")
    if targets[-1] > sample_count:
        raise ValueError("largest target exceeds sample count.")
    return targets
