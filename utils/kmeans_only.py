from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans

from utils.checkpoint_pc_hbm import normalize_sample_key


KMEANS_ONLY_PROTOCOL_VERSION = "kmeans_only_sqrt_quota_center_distance_v1"


@dataclass(frozen=True)
class KMeansOnlyFit:
    cluster_ids: torch.Tensor
    center_distances: torch.Tensor
    seed_keys: tuple[str, ...]
    centers: torch.Tensor


@dataclass(frozen=True)
class KMeansOnlyRound:
    target_count: int
    budget: int
    quotas: dict[int, int]


@dataclass(frozen=True)
class KMeansOnlySelectionResult:
    seed_keys: tuple[str, ...]
    splits: dict[int, list[str]]
    selection_order: tuple[str, ...]
    selection_rank: dict[int, dict[str, int]]
    first_selected_target: dict[str, int]
    rounds: tuple[KMeansOnlyRound, ...]
    protocol_version: str = KMEANS_ONLY_PROTOCOL_VERSION


def normalize_kmeans_features(
    features: Any,
    *,
    expected_length: int | None = None,
) -> torch.Tensor:
    """Validate and L2-normalize a real feature matrix on CPU in float32."""

    try:
        tensor = torch.as_tensor(features).detach().cpu()
    except (TypeError, ValueError, RuntimeError) as error:
        raise TypeError("features must be convertible to a numeric tensor.") from error
    if tensor.ndim != 2 or tensor.shape[1] == 0:
        raise ValueError("features must have shape [N, D] with D > 0.")
    if expected_length is not None and tensor.shape[0] != expected_length:
        raise ValueError(
            f"features must have {expected_length} rows aligned to sample_keys."
        )
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
        raise ValueError(
            f"features contain zero-norm rows: {zero_rows[:3].tolist()!r}"
        )
    normalized = tensor / norms.unsqueeze(1)
    if not bool(torch.isfinite(normalized).all()):
        raise RuntimeError("normalized features unexpectedly contain NaN or infinity.")
    return normalized.contiguous()


def fit_kmeans_only(
    sample_keys: Sequence[str],
    features: Any,
    *,
    n_clusters: int = 40,
    random_seed: int = 2025,
) -> KMeansOnlyFit:
    """Fit deterministic KMeans and select one center-nearest seed per cluster."""

    keys = _normalize_unique_keys(sample_keys, name="sample_keys")
    normalized = normalize_kmeans_features(features, expected_length=len(keys))
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, Integral):
        raise TypeError("n_clusters must be an integer.")
    n_clusters = int(n_clusters)
    if not 0 < n_clusters <= len(keys):
        raise ValueError("n_clusters must be in [1, sample_count].")
    if isinstance(random_seed, bool) or not isinstance(random_seed, Integral):
        raise TypeError("random_seed must be an integer.")

    matrix = normalized.numpy()
    unique_count = int(np.unique(matrix, axis=0).shape[0])
    if unique_count < n_clusters:
        raise ValueError(
            f"KMeans requires at least {n_clusters} distinct features; "
            f"found {unique_count}."
        )
    model = KMeans(
        n_clusters=n_clusters,
        random_state=int(random_seed),
        n_init=10,
        algorithm="lloyd",
    ).fit(matrix)
    cluster_ids = model.labels_.astype(np.int64, copy=False)

    # Recompute centers with catalog-ordered float64 accumulation.  This keeps
    # center distances stable across BLAS implementations while preserving the
    # fitted labels from sklearn.
    matrix64 = matrix.astype(np.float64, copy=False)
    centers64 = np.empty((n_clusters, matrix.shape[1]), dtype=np.float64)
    for cluster_id in range(n_clusters):
        members = np.flatnonzero(cluster_ids == cluster_id)
        if members.size == 0:
            raise RuntimeError(f"KMeans produced empty cluster {cluster_id}.")
        center = np.zeros(matrix.shape[1], dtype=np.float64)
        for member_index in members.tolist():
            center += matrix64[member_index]
        centers64[cluster_id] = center / float(members.size)
    residuals = matrix64 - centers64[cluster_ids]
    distances = np.sqrt(np.sum(residuals * residuals, axis=1, dtype=np.float64))
    distances32 = distances.astype(np.float32, copy=False)

    seed_keys: list[str] = []
    for cluster_id in range(n_clusters):
        members = np.flatnonzero(cluster_ids == cluster_id).tolist()
        seed_index = min(
            members,
            key=lambda index: (float(distances32[index]), keys[index]),
        )
        seed_keys.append(keys[seed_index])

    return KMeansOnlyFit(
        cluster_ids=torch.from_numpy(cluster_ids.copy()).long(),
        center_distances=torch.from_numpy(distances32.copy()),
        seed_keys=tuple(seed_keys),
        centers=torch.from_numpy(centers64.astype(np.float32, copy=False).copy()),
    )


def allocate_sqrt_quotas(
    cluster_capacities: Mapping[int, int],
    budget: int,
) -> dict[int, int]:
    """Allocate an exact budget by sqrt(capacity) and largest remainders."""

    if isinstance(budget, bool) or not isinstance(budget, Integral):
        raise TypeError("budget must be an integer.")
    budget = int(budget)
    if budget < 0:
        raise ValueError("budget must be non-negative.")

    capacities: dict[int, int] = {}
    for raw_cluster_id, raw_capacity in cluster_capacities.items():
        if isinstance(raw_cluster_id, bool) or not isinstance(raw_cluster_id, Integral):
            raise TypeError("cluster ids must be integers.")
        if isinstance(raw_capacity, bool) or not isinstance(raw_capacity, Integral):
            raise TypeError("cluster capacities must be integers.")
        cluster_id = int(raw_cluster_id)
        capacity = int(raw_capacity)
        if cluster_id in capacities:
            raise ValueError("cluster ids are ambiguous after integer normalization.")
        if capacity < 0:
            raise ValueError(f"cluster {cluster_id} has negative capacity.")
        capacities[cluster_id] = capacity
    if not capacities:
        if budget:
            raise ValueError("budget exceeds empty cluster capacity.")
        return {}
    if budget > sum(capacities.values()):
        raise ValueError("budget exceeds remaining cluster capacity.")
    if budget == 0:
        return {cluster_id: 0 for cluster_id in sorted(capacities)}

    active = {
        cluster_id: capacity
        for cluster_id, capacity in capacities.items()
        if capacity > 0
    }
    weights = {
        cluster_id: math.sqrt(capacity)
        for cluster_id, capacity in active.items()
    }
    weight_sum = sum(weights.values())
    raw = {
        cluster_id: budget * weights[cluster_id] / weight_sum
        for cluster_id in active
    }
    quotas = {
        cluster_id: min(active[cluster_id], math.floor(raw[cluster_id]))
        for cluster_id in active
    }
    remaining = budget - sum(quotas.values())
    remainder_order = sorted(
        active,
        key=lambda cluster_id: (
            -(raw[cluster_id] - math.floor(raw[cluster_id])),
            cluster_id,
        ),
    )
    while remaining:
        progress = False
        for cluster_id in remainder_order:
            if remaining == 0:
                break
            if quotas[cluster_id] >= active[cluster_id]:
                continue
            quotas[cluster_id] += 1
            remaining -= 1
            progress = True
        if not progress:
            raise RuntimeError("Unable to allocate the exact KMeans-only budget.")

    result = {
        cluster_id: quotas.get(cluster_id, 0)
        for cluster_id in sorted(capacities)
    }
    if sum(result.values()) != budget:
        raise RuntimeError("KMeans-only quota allocation missed the exact budget.")
    if any(result[cid] > capacities[cid] for cid in result):
        raise RuntimeError("KMeans-only quota allocation exceeded capacity.")
    return result


def build_kmeans_only_nested_splits(
    sample_keys: Sequence[str],
    cluster_ids: Sequence[int] | np.ndarray | torch.Tensor,
    center_distances: Sequence[float] | np.ndarray | torch.Tensor,
    seed_keys: Sequence[str],
    target_counts: Sequence[int] = (41, 202, 404, 808),
) -> KMeansOnlySelectionResult:
    """Build exact nested splits using only cluster quotas and center distance."""

    keys = _normalize_unique_keys(sample_keys, name="sample_keys")
    clusters = _validated_cluster_ids(cluster_ids, expected_length=len(keys))
    distances = _validated_distances(center_distances, expected_length=len(keys))
    seeds = _normalize_unique_keys(seed_keys, name="seed_keys")
    targets = _validated_targets(target_counts, sample_count=len(keys))

    key_to_index = {key: index for index, key in enumerate(keys)}
    missing_seeds = sorted(set(seeds) - set(key_to_index))
    if missing_seeds:
        raise ValueError(f"seed_keys are absent from the catalog: {missing_seeds[:3]!r}")
    cluster_count = int(clusters.max().item()) + 1
    expected_cluster_ids = set(range(cluster_count))
    actual_cluster_ids = set(int(value) for value in clusters.tolist())
    if actual_cluster_ids != expected_cluster_ids:
        raise ValueError("cluster_ids must be contiguous and start at zero.")
    if len(seeds) != cluster_count:
        raise ValueError("seed_keys must contain exactly one key per cluster.")
    seed_clusters = [int(clusters[key_to_index[key]]) for key in seeds]
    if set(seed_clusters) != expected_cluster_ids:
        raise ValueError("seed_keys must cover every cluster exactly once.")
    if len(seeds) > targets[0]:
        raise ValueError("seed count exceeds the smallest target.")

    for seed in seeds:
        index = key_to_index[seed]
        cluster_id = int(clusters[index])
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

    seed_by_cluster = {
        int(clusters[key_to_index[key]]): key
        for key in seeds
    }
    ordered_seeds = [seed_by_cluster[cluster_id] for cluster_id in range(cluster_count)]
    selected = set(ordered_seeds)
    selection_order = list(ordered_seeds)
    first_selected_target = {key: len(ordered_seeds) for key in ordered_seeds}
    splits: dict[int, list[str]] = {}
    rounds: list[KMeansOnlyRound] = []
    previous: set[str] | None = None

    for target in targets:
        budget = target - len(selected)
        capacities = Counter(
            int(clusters[key_to_index[key]])
            for key in keys
            if key not in selected
        )
        quotas = allocate_sqrt_quotas(
            {
                cluster_id: capacities.get(cluster_id, 0)
                for cluster_id in range(cluster_count)
            },
            budget,
        )
        round_added: list[str] = []
        for cluster_id in range(cluster_count):
            candidates = sorted(
                (
                    key
                    for key in keys
                    if key not in selected
                    and int(clusters[key_to_index[key]]) == cluster_id
                ),
                key=lambda key: (
                    float(distances[key_to_index[key]]),
                    key,
                ),
            )
            chosen = candidates[: quotas[cluster_id]]
            if len(chosen) != quotas[cluster_id]:
                raise RuntimeError(
                    f"Cluster {cluster_id} could not satisfy quota {quotas[cluster_id]}."
                )
            round_added.extend(chosen)

        if len(round_added) != budget or len(round_added) != len(set(round_added)):
            raise RuntimeError("KMeans-only round did not produce its exact unique budget.")
        for key in round_added:
            selected.add(key)
            selection_order.append(key)
            first_selected_target[key] = target
        if len(selected) != target:
            raise RuntimeError(
                f"KMeans-only selection produced {len(selected)} samples for {target}."
            )
        if previous is not None and not previous.issubset(selected):
            raise RuntimeError("Nested split invariant was violated.")
        splits[target] = sorted(selected)
        rounds.append(
            KMeansOnlyRound(
                target_count=target,
                budget=budget,
                quotas=quotas,
            )
        )
        previous = set(selected)

    overall_rank = {
        key: rank for rank, key in enumerate(selection_order, start=1)
    }
    return KMeansOnlySelectionResult(
        seed_keys=tuple(ordered_seeds),
        splits=splits,
        selection_order=tuple(selection_order),
        selection_rank={
            target: {key: overall_rank[key] for key in split}
            for target, split in splits.items()
        },
        first_selected_target=first_selected_target,
        rounds=tuple(rounds),
    )


def labeled_names_from_keys(sample_keys: Sequence[str]) -> list[str]:
    keys = _normalize_unique_keys(sample_keys, name="sample_keys")
    names = [key.rsplit("/", 1)[-1] for key in keys]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise ValueError(f"sample keys produce duplicate labeled stems: {duplicates[:3]!r}")
    return names


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
    try:
        tensor = torch.as_tensor(value).detach().cpu()
    except (TypeError, ValueError, RuntimeError) as error:
        raise TypeError("cluster_ids must be convertible to a tensor.") from error
    if tensor.ndim != 1 or tensor.numel() != expected_length:
        raise ValueError(f"cluster_ids must contain {expected_length} elements.")
    if tensor.dtype == torch.bool or torch.is_complex(tensor):
        raise TypeError("cluster_ids must contain integers.")
    integer = tensor.to(dtype=torch.int64)
    if not torch.equal(tensor, integer.to(dtype=tensor.dtype)):
        raise ValueError("cluster_ids must contain exact integers.")
    if bool((integer < 0).any()):
        raise ValueError("cluster_ids must be non-negative.")
    return integer.contiguous()


def _validated_distances(value: Any, *, expected_length: int) -> torch.Tensor:
    try:
        tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu().contiguous()
    except (TypeError, ValueError, RuntimeError) as error:
        raise TypeError("center_distances must be convertible to a numeric tensor.") from error
    if tensor.ndim != 1 or tensor.numel() != expected_length:
        raise ValueError(f"center_distances must contain {expected_length} elements.")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError("center_distances contain NaN or infinity.")
    if bool((tensor < 0.0).any()):
        raise ValueError("center_distances must be non-negative.")
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
    "KMEANS_ONLY_PROTOCOL_VERSION",
    "KMeansOnlyFit",
    "KMeansOnlyRound",
    "KMeansOnlySelectionResult",
    "allocate_sqrt_quotas",
    "build_kmeans_only_nested_splits",
    "fit_kmeans_only",
    "labeled_names_from_keys",
    "normalize_kmeans_features",
]
