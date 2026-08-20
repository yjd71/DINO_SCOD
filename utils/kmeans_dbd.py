from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
import torch

from utils.checkpoint_pc_hbm import normalize_sample_key
from utils.kmeans_only import allocate_sqrt_quotas, normalize_kmeans_features


KMEANS_DBD_DEDUP_PROTOCOL_VERSION = "kmeans_dbd_incluster_dino_dedup_v1"


@dataclass(frozen=True)
class KMeansDBDDecision:
    decision: str
    skip_count: int
    max_cosine_similarity: float | None
    reference_key: str | None
    first_selected_target: int | None
    relaxed_backfill: bool


@dataclass(frozen=True)
class KMeansDBDRound:
    target_count: int
    budget: int
    quotas: dict[int, int]
    quota_selected_count: int
    dedup_backfill_count: int
    relaxed_backfill_count: int
    dedup_skips: int


@dataclass(frozen=True)
class KMeansDBDSelectionResult:
    seed_keys: tuple[str, ...]
    splits: dict[int, list[str]]
    selection_order: tuple[str, ...]
    selection_rank: dict[int, dict[str, int]]
    first_selected_target: dict[str, int]
    decisions: dict[str, KMeansDBDDecision]
    rounds: tuple[KMeansDBDRound, ...]
    dedup_threshold: float
    protocol_version: str = KMEANS_DBD_DEDUP_PROTOCOL_VERSION


def build_kmeans_dbd_nested_splits(
    sample_keys: Sequence[str],
    features: Any,
    cluster_ids: Sequence[int] | np.ndarray | torch.Tensor,
    center_distances: Sequence[float] | np.ndarray | torch.Tensor,
    boundary_disagreement: Sequence[float] | np.ndarray | torch.Tensor,
    seed_keys: Sequence[str],
    target_counts: Sequence[int] = (41, 202, 404, 808),
    dedup_threshold: float = 0.95,
) -> KMeansDBDSelectionResult:
    """Select nested KMeans splits ranked only by boundary disagreement.

    KMeans supplies cluster membership, one center-nearest seed per cluster,
    and square-root quotas.  After the seeds, ``boundary_disagreement`` is the
    only ranking value.  DINO features are used solely for same-cluster cosine
    deduplication.
    """

    keys = _normalize_unique_keys(sample_keys, name="sample_keys")
    normalized_features = normalize_kmeans_features(
        features,
        expected_length=len(keys),
    )
    clusters = _validated_cluster_ids(cluster_ids, expected_length=len(keys))
    distances = _validated_vector(
        center_distances,
        expected_length=len(keys),
        name="center_distances",
        lower_bound=0.0,
    )
    boundary = _validated_vector(
        boundary_disagreement,
        expected_length=len(keys),
        name="boundary_disagreement",
        lower_bound=0.0,
        upper_bound=1.0,
    )
    seeds = _normalize_unique_keys(seed_keys, name="seed_keys")
    targets = _validated_targets(target_counts, sample_count=len(keys))

    if not math.isfinite(dedup_threshold) or not 0.0 < dedup_threshold <= 1.0:
        raise ValueError("dedup_threshold must be finite and in (0, 1].")

    key_to_index = {key: index for index, key in enumerate(keys)}
    missing_seeds = sorted(set(seeds) - set(key_to_index))
    if missing_seeds:
        raise ValueError(f"seed_keys are absent from the catalog: {missing_seeds[:3]!r}")

    cluster_count = int(clusters.max().item()) + 1
    expected_clusters = set(range(cluster_count))
    actual_clusters = set(int(value) for value in clusters.tolist())
    if actual_clusters != expected_clusters:
        raise ValueError("cluster_ids must be contiguous and start at zero.")
    if len(seeds) != cluster_count:
        raise ValueError("seed_keys must contain exactly one key per cluster.")
    if len(seeds) >= targets[0]:
        raise ValueError("the smallest target must exceed the seed count.")

    seed_by_cluster: dict[int, str] = {}
    for seed in seeds:
        seed_index = key_to_index[seed]
        cluster_id = int(clusters[seed_index])
        if cluster_id in seed_by_cluster:
            raise ValueError("seed_keys must cover every cluster exactly once.")
        nearest = min(
            (
                key
                for key in keys
                if int(clusters[key_to_index[key]]) == cluster_id
            ),
            key=lambda key: (float(distances[key_to_index[key]]), key),
        )
        if seed != nearest:
            raise ValueError(
                f"seed {seed!r} is not the center-nearest key for cluster {cluster_id}."
            )
        seed_by_cluster[cluster_id] = seed
    if set(seed_by_cluster) != expected_clusters:
        raise ValueError("seed_keys must cover every cluster exactly once.")

    ranked_by_cluster: dict[int, list[int]] = {
        cluster_id: [] for cluster_id in range(cluster_count)
    }
    for index, cluster_id in enumerate(clusters.tolist()):
        ranked_by_cluster[int(cluster_id)].append(index)
    for indices in ranked_by_cluster.values():
        indices.sort(key=lambda index: (-float(boundary[index]), keys[index]))

    ordered_seeds = tuple(sorted(seed_by_cluster.values()))
    selected = set(ordered_seeds)
    selection_order = list(ordered_seeds)
    selected_by_cluster: dict[int, list[int]] = {
        cluster_id: [] for cluster_id in range(cluster_count)
    }
    for seed in ordered_seeds:
        index = key_to_index[seed]
        selected_by_cluster[int(clusters[index])].append(index)

    mutable_decisions: dict[str, dict[str, Any]] = {
        key: {
            "decision": "not_evaluated",
            "skip_count": 0,
            "max_cosine_similarity": None,
            "reference_key": None,
            "first_selected_target": None,
            "relaxed_backfill": False,
        }
        for key in keys
    }
    first_selected_target = {key: len(ordered_seeds) for key in ordered_seeds}
    for seed in ordered_seeds:
        mutable_decisions[seed].update(
            decision="seed",
            first_selected_target=len(ordered_seeds),
        )

    def duplicate_reference(index: int) -> tuple[bool, float | None, str | None]:
        cluster_id = int(clusters[index])
        references = selected_by_cluster[cluster_id]
        if not references:
            return False, None, None
        similarities = normalized_features[references] @ normalized_features[index]
        maximum = float(similarities.max())
        tied_references = [
            references[offset]
            for offset, value in enumerate(similarities.tolist())
            if value == maximum
        ]
        reference_key = min(keys[reference] for reference in tied_references)
        decision = mutable_decisions[keys[index]]
        previous_maximum = decision["max_cosine_similarity"]
        previous_reference = decision["reference_key"]
        if (
            previous_maximum is None
            or maximum > previous_maximum
            or (maximum == previous_maximum and reference_key < previous_reference)
        ):
            decision["max_cosine_similarity"] = maximum
            decision["reference_key"] = reference_key
        return maximum > dedup_threshold, maximum, reference_key

    def accept(index: int, *, target: int, stage: str) -> None:
        key = keys[index]
        if key in selected:
            raise RuntimeError(f"attempted to select duplicate key {key!r}.")
        selected.add(key)
        selection_order.append(key)
        selected_by_cluster[int(clusters[index])].append(index)
        first_selected_target[key] = target
        mutable_decisions[key].update(
            decision=stage,
            first_selected_target=target,
            relaxed_backfill=stage == "relaxed_backfill",
        )

    def reject_duplicate(index: int) -> None:
        decision = mutable_decisions[keys[index]]
        decision["skip_count"] += 1
        if decision["decision"] == "not_evaluated":
            decision["decision"] = "dedup_skipped"

    splits: dict[int, list[str]] = {}
    rounds: list[KMeansDBDRound] = []
    previous_split: set[str] | None = None
    for target in targets:
        budget = target - len(selected)
        capacities = Counter(
            int(clusters[index])
            for index, key in enumerate(keys)
            if key not in selected
        )
        quotas = allocate_sqrt_quotas(
            {
                cluster_id: capacities.get(cluster_id, 0)
                for cluster_id in range(cluster_count)
            },
            budget,
        )

        quota_selected = 0
        dedup_backfill = 0
        relaxed_backfill = 0
        dedup_skips = 0
        for cluster_id in range(cluster_count):
            accepted_for_cluster = 0
            for index in ranked_by_cluster[cluster_id]:
                if accepted_for_cluster >= quotas[cluster_id]:
                    break
                if keys[index] in selected:
                    continue
                duplicate, _, _ = duplicate_reference(index)
                if duplicate:
                    reject_duplicate(index)
                    dedup_skips += 1
                    continue
                accept(index, target=target, stage="quota")
                accepted_for_cluster += 1
                quota_selected += 1

        if len(selected) < target:
            global_remaining = sorted(
                (index for index, key in enumerate(keys) if key not in selected),
                key=lambda index: (-float(boundary[index]), keys[index]),
            )
            for index in global_remaining:
                if len(selected) >= target:
                    break
                duplicate, _, _ = duplicate_reference(index)
                if duplicate:
                    reject_duplicate(index)
                    dedup_skips += 1
                    continue
                accept(index, target=target, stage="dedup_backfill")
                dedup_backfill += 1

        if len(selected) < target:
            global_remaining = sorted(
                (index for index, key in enumerate(keys) if key not in selected),
                key=lambda index: (-float(boundary[index]), keys[index]),
            )
            for index in global_remaining:
                if len(selected) >= target:
                    break
                accept(index, target=target, stage="relaxed_backfill")
                relaxed_backfill += 1

        if len(selected) != target:
            raise RuntimeError(
                f"KMeans-D_bd selected {len(selected)} samples for target {target}."
            )
        current_split = set(selected)
        if previous_split is not None and not previous_split.issubset(current_split):
            raise RuntimeError("nested split invariant was violated.")
        splits[target] = sorted(current_split)
        rounds.append(
            KMeansDBDRound(
                target_count=target,
                budget=budget,
                quotas=quotas,
                quota_selected_count=quota_selected,
                dedup_backfill_count=dedup_backfill,
                relaxed_backfill_count=relaxed_backfill,
                dedup_skips=dedup_skips,
            )
        )
        previous_split = current_split

    overall_rank = {
        key: rank for rank, key in enumerate(selection_order, start=1)
    }
    decisions = {
        key: KMeansDBDDecision(**mutable_decisions[key]) for key in keys
    }
    return KMeansDBDSelectionResult(
        seed_keys=ordered_seeds,
        splits=splits,
        selection_order=tuple(selection_order),
        selection_rank={
            target: {key: overall_rank[key] for key in split}
            for target, split in splits.items()
        },
        first_selected_target=first_selected_target,
        decisions=decisions,
        rounds=tuple(rounds),
        dedup_threshold=float(dedup_threshold),
    )


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


def _validated_cluster_ids(value: Any, *, expected_length: int) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim != 1 or tensor.numel() != expected_length:
        raise ValueError(f"cluster_ids must contain {expected_length} elements.")
    if tensor.dtype == torch.bool or tensor.is_floating_point() or torch.is_complex(tensor):
        raise TypeError("cluster_ids must contain integers.")
    tensor = tensor.detach().cpu().long().contiguous()
    if bool((tensor < 0).any()):
        raise ValueError("cluster_ids must be non-negative.")
    return tensor


def _validated_vector(
    value: Any,
    *,
    expected_length: int,
    name: str,
    lower_bound: float,
    upper_bound: float | None = None,
) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if tensor.ndim != 1 or tensor.numel() != expected_length:
        raise ValueError(f"{name} must contain {expected_length} elements.")
    if tensor.dtype == torch.bool or torch.is_complex(tensor):
        raise TypeError(f"{name} must contain real numeric values.")
    tensor = tensor.detach().cpu().to(dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} contains NaN or infinity.")
    if bool((tensor < lower_bound).any()):
        raise ValueError(f"{name} contains values below {lower_bound}.")
    if upper_bound is not None and bool((tensor > upper_bound).any()):
        raise ValueError(f"{name} contains values above {upper_bound}.")
    return tensor


def _validated_targets(
    target_counts: Sequence[int],
    *,
    sample_count: int,
) -> tuple[int, ...]:
    if isinstance(target_counts, (str, bytes)) or not isinstance(target_counts, Sequence):
        raise TypeError("target_counts must be a sequence of integers.")
    if not target_counts:
        raise ValueError("target_counts must not be empty.")
    if any(isinstance(value, bool) or not isinstance(value, Integral) for value in target_counts):
        raise TypeError("target_counts must contain integers.")
    targets = tuple(int(value) for value in target_counts)
    if any(value <= 0 for value in targets):
        raise ValueError("target_counts must be positive.")
    if tuple(sorted(set(targets))) != targets:
        raise ValueError("target_counts must be strictly increasing.")
    if targets[-1] > sample_count:
        raise ValueError("largest target exceeds sample count.")
    return targets


__all__ = [
    "KMEANS_DBD_DEDUP_PROTOCOL_VERSION",
    "KMeansDBDDecision",
    "KMeansDBDRound",
    "KMeansDBDSelectionResult",
    "build_kmeans_dbd_nested_splits",
]
