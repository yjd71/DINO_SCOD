from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import __version__ as sklearn_version
from sklearn.cluster import KMeans

from configs.pc_bacs_config import PCBACSConfig, SCORE_FORMULA_VERSION
from utils.checkpoint_pc_hbm import (
    compute_labeled_split_fingerprint,
    normalize_sample_key,
)


FEATURE_CACHE_FORMAT_VERSION = 1
SCORE_CACHE_FORMAT_VERSION = 1
MANIFEST_FORMAT_VERSION = 1
FEATURE_TYPE = "dinov2_vitb14_global"
PREPROCESSING_VERSION = "rgb_392_bilinear_antialias_imagenet_v1"


@dataclass(frozen=True)
class KMeansResult:
    cluster_ids: torch.Tensor
    center_distances: torch.Tensor
    seed_keys: list[str]
    normalized_features: torch.Tensor
    centers: torch.Tensor


@dataclass(frozen=True)
class NestedSelectionResult:
    splits: dict[int, list[str]]
    selection_order: tuple[str, ...]
    selection_rank: dict[int, dict[str, int]]
    dedup_skipped_count: dict[str, int]
    rounds: tuple[dict[str, Any], ...]


def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    kernel_x = torch.tensor(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
        device=device,
        dtype=dtype,
    ) / 8.0
    kernel_y = kernel_x.transpose(0, 1)
    return kernel_x.view(1, 1, 3, 3), kernel_y.view(1, 1, 3, 3)


def _validate_probability_shape(probability: torch.Tensor, name: str) -> None:
    if not isinstance(probability, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if probability.ndim != 4 or probability.shape[1] != 1:
        raise ValueError(f"Expected {name} [B,1,H,W], got {tuple(probability.shape)}")
    if probability.shape[0] <= 0 or probability.shape[2] <= 0 or probability.shape[3] <= 0:
        raise ValueError(f"{name} must have non-empty batch and spatial dimensions.")


def sobel_magnitude(probability: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Return the Sobel magnitude with replicate padding.

    ``eps`` is accepted for API compatibility and validated here, but is not
    injected into the magnitude.  PC-BACS uses epsilon only in the weighted
    disagreement denominator so a constant prediction has exactly zero edge
    weight.
    """

    _validate_probability_shape(probability, "probability")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive.")

    probability = torch.nan_to_num(
        probability.float(), nan=0.0, posinf=1.0, neginf=0.0
    )
    kernel_x, kernel_y = _sobel_kernels(probability.device, probability.dtype)
    padded = F.pad(probability, (1, 1, 1, 1), mode="replicate")
    grad_x = F.conv2d(padded, kernel_x)
    grad_y = F.conv2d(padded, kernel_y)
    return torch.hypot(grad_x, grad_y)


def compute_pc_bacs_score(
    probability: torch.Tensor,
    transformed_probability: torch.Tensor,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Compute boundary disagreement, global disagreement and PC-BACS value."""

    _validate_probability_shape(probability, "probability")
    _validate_probability_shape(transformed_probability, "transformed_probability")
    if probability.shape != transformed_probability.shape:
        raise ValueError("Two prediction views must have identical shapes.")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive.")

    probability = torch.nan_to_num(
        probability.float(), nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    transformed_probability = torch.nan_to_num(
        transformed_probability.float(), nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)

    mean_probability = 0.5 * (probability + transformed_probability)
    difference = (probability - transformed_probability).abs()
    boundary_weight = sobel_magnitude(mean_probability, eps=eps)

    boundary_num = (boundary_weight * difference).flatten(1).sum(dim=1)
    boundary_den = boundary_weight.flatten(1).sum(dim=1).clamp_min(eps)
    boundary_disagreement = torch.nan_to_num(
        boundary_num / boundary_den, nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    global_disagreement = torch.nan_to_num(
        difference.flatten(1).mean(dim=1), nan=0.0, posinf=1.0, neginf=0.0
    ).clamp(0.0, 1.0)
    score = torch.nan_to_num(
        boundary_disagreement * (1.0 - global_disagreement),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    ).clamp(0.0, 1.0)

    return {
        "boundary_disagreement": boundary_disagreement,
        "global_disagreement": global_disagreement,
        "score": score,
    }


@torch.inference_mode()
def score_pool(
    selector,
    loader,
    device: str | torch.device,
    use_amp: bool = True,
    eps: float = 1e-6,
) -> list[dict[str, Any]]:
    """Score a deterministic RGB-only loader through the legacy Decoder path."""

    device = torch.device(device)
    selector.eval()
    records: list[dict[str, Any]] = []

    for sample_keys, images in loader:
        if len(sample_keys) != int(images.shape[0]):
            raise ValueError("Loader sample key count does not match image batch size.")
        images = images.to(device, non_blocking=True)
        two_views = torch.cat([images, torch.flip(images, dims=[-1])], dim=0)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bool(use_amp and device.type == "cuda"),
        ):
            outputs = selector(two_views, pc_mode="off")
            if not isinstance(outputs, (tuple, list)) or len(outputs) <= 3:
                raise ValueError("Selector must return a sequence containing logits at index 3.")
            logits = outputs[3]

        if not isinstance(logits, torch.Tensor) or logits.shape[0] != two_views.shape[0]:
            raise ValueError("Selector logits have an invalid batch dimension.")
        probability, transformed_probability = logits.sigmoid().chunk(2, dim=0)
        transformed_probability = torch.flip(transformed_probability, dims=[-1])
        score_dict = compute_pc_bacs_score(
            probability, transformed_probability, eps=eps
        )

        for index, key in enumerate(sample_keys):
            records.append(
                {
                    "sample_key": normalize_sample_key(str(key)),
                    "boundary_disagreement": float(
                        score_dict["boundary_disagreement"][index].cpu()
                    ),
                    "global_disagreement": float(
                        score_dict["global_disagreement"][index].cpu()
                    ),
                    "score": float(score_dict["score"][index].cpu()),
                }
            )

    keys = [record["sample_key"] for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Scoring loader produced duplicate sample keys.")
    return records


def normalize_dino_features(features: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if not isinstance(features, torch.Tensor):
        features = torch.as_tensor(features)
    if features.ndim != 2 or features.shape[0] <= 0 or features.shape[1] <= 0:
        raise ValueError(f"Expected features [N,D], got {tuple(features.shape)}")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive.")
    features = features.detach().float().cpu().contiguous()
    if not torch.isfinite(features).all():
        raise ValueError("DINO features contain NaN or infinity.")
    norms = torch.linalg.vector_norm(features, dim=1)
    if bool((norms <= eps).any()):
        raise ValueError("DINO features contain a zero-norm row.")
    return F.normalize(features, p=2.0, dim=1, eps=eps)


def fit_dino_kmeans(
    sample_keys: Sequence[str],
    features: torch.Tensor,
    n_clusters: int = 40,
    random_seed: int = 2025,
) -> KMeansResult:
    """Fit the frozen KMeans protocol and choose one center-nearest seed per cluster."""

    keys = _normalize_unique_keys(sample_keys, name="sample_keys")
    normalized = normalize_dino_features(features)
    if len(keys) != normalized.shape[0]:
        raise ValueError("sample key count does not match feature row count.")
    if isinstance(n_clusters, bool) or not isinstance(n_clusters, int) or n_clusters <= 0:
        raise ValueError("n_clusters must be a positive integer.")
    if n_clusters > len(keys):
        raise ValueError("n_clusters exceeds sample count.")

    matrix = normalized.numpy()
    unique_count = int(np.unique(matrix, axis=0).shape[0])
    if unique_count < n_clusters:
        raise ValueError(
            f"KMeans requires at least {n_clusters} distinct features; found {unique_count}."
        )
    model = KMeans(
        n_clusters=n_clusters,
        random_state=int(random_seed),
        n_init=10,
        algorithm="lloyd",
    ).fit(matrix)
    cluster_ids_np = model.labels_.astype(np.int64, copy=False)
    # Recompute the fitted centers from the final labels with an explicit,
    # catalog-ordered float64 accumulation.  BLAS-backed float32 reductions can
    # otherwise vary by one ULP between identical Windows processes, which is
    # enough to make the audit CSV differ even though labels and selections are
    # identical.
    matrix64 = matrix.astype(np.float64, copy=False)
    canonical_centers = np.empty((n_clusters, matrix.shape[1]), dtype=np.float64)
    for cluster_id in range(n_clusters):
        members = np.flatnonzero(cluster_ids_np == cluster_id)
        if members.size == 0:
            raise RuntimeError(f"KMeans produced empty cluster {cluster_id}.")
        center = np.zeros(matrix.shape[1], dtype=np.float64)
        for member_index in members.tolist():
            center += matrix64[member_index]
        canonical_centers[cluster_id] = center / float(members.size)
    residuals = matrix64 - canonical_centers[cluster_ids_np]
    distances_np = np.sqrt(np.sum(residuals * residuals, axis=1, dtype=np.float64))

    seed_keys: list[str] = []
    for cluster_id in range(n_clusters):
        members = np.flatnonzero(cluster_ids_np == cluster_id).tolist()
        if not members:
            raise RuntimeError(f"KMeans produced empty cluster {cluster_id}.")
        seed_index = min(members, key=lambda index: (float(distances_np[index]), keys[index]))
        seed_keys.append(keys[seed_index])

    return KMeansResult(
        cluster_ids=torch.from_numpy(cluster_ids_np.copy()).long(),
        center_distances=torch.from_numpy(distances_np.astype(np.float32, copy=False).copy()),
        seed_keys=seed_keys,
        normalized_features=normalized,
        centers=torch.from_numpy(canonical_centers.astype(np.float32, copy=False).copy()),
    )


def allocate_cluster_quotas(cluster_sizes: Mapping[int, int], budget: int) -> dict[int, int]:
    """Allocate an exact square-root-weighted budget with largest remainders."""

    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a non-negative integer.")
    sizes: dict[int, int] = {}
    for cluster_id, size in cluster_sizes.items():
        if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or int(size) < 0:
            raise ValueError(f"cluster {cluster_id} has invalid capacity {size!r}.")
        sizes[int(cluster_id)] = int(size)
    if len(sizes) != len(cluster_sizes):
        raise ValueError("cluster ids are ambiguous after integer normalization.")
    total_capacity = sum(sizes.values())
    if budget > total_capacity:
        raise ValueError("budget exceeds remaining candidate capacity.")
    if budget == 0:
        return {cluster_id: 0 for cluster_id in sorted(sizes)}

    active = {cluster_id: size for cluster_id, size in sizes.items() if size > 0}
    weights = {cluster_id: math.sqrt(size) for cluster_id, size in active.items()}
    weight_sum = sum(weights.values())
    raw = {
        cluster_id: budget * weights[cluster_id] / weight_sum for cluster_id in active
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
            raise RuntimeError("Unable to allocate exact cluster budget.")

    result = {cluster_id: quotas.get(cluster_id, 0) for cluster_id in sorted(sizes)}
    if sum(result.values()) != budget:
        raise RuntimeError("Cluster quota allocation did not meet the exact budget.")
    if any(result[cluster_id] > sizes[cluster_id] for cluster_id in result):
        raise RuntimeError("Cluster quota allocation exceeded capacity.")
    return result


def build_nested_splits(
    sample_keys: Sequence[str],
    features: torch.Tensor,
    cluster_ids: Sequence[int] | np.ndarray | torch.Tensor,
    scores: Sequence[float] | np.ndarray | torch.Tensor,
    seed_keys: Sequence[str],
    target_counts: Sequence[int] = (41, 202, 404),
    dedup_threshold: float = 0.98,
) -> NestedSelectionResult:
    """Build exact nested splits using quotas, in-cluster dedup and three-level fill."""

    keys = _normalize_unique_keys(sample_keys, name="sample_keys")
    normalized_features = normalize_dino_features(features)
    sample_count = len(keys)
    if normalized_features.shape[0] != sample_count:
        raise ValueError("sample key count does not match feature row count.")

    cluster_tensor = torch.as_tensor(cluster_ids)
    if cluster_tensor.ndim != 1 or cluster_tensor.numel() != sample_count:
        raise ValueError("cluster_ids must be a one-dimensional vector aligned to sample_keys.")
    if cluster_tensor.dtype == torch.bool or cluster_tensor.is_floating_point():
        raise TypeError("cluster_ids must contain integers.")
    cluster_tensor = cluster_tensor.detach().cpu().long()
    if bool((cluster_tensor < 0).any()):
        raise ValueError("cluster_ids must be non-negative.")

    score_tensor = torch.as_tensor(scores, dtype=torch.float32).detach().cpu()
    if score_tensor.ndim != 1 or score_tensor.numel() != sample_count:
        raise ValueError("scores must be a one-dimensional vector aligned to sample_keys.")
    if not torch.isfinite(score_tensor).all():
        raise ValueError("scores contain NaN or infinity.")

    targets = tuple(target_counts)
    if not targets:
        raise ValueError("target_counts must not be empty.")
    if any(isinstance(target, bool) or not isinstance(target, (int, np.integer)) for target in targets):
        raise TypeError("target_counts must contain integers.")
    targets = tuple(int(target) for target in targets)
    if any(target <= 0 for target in targets):
        raise ValueError("target_counts must be positive.")
    if tuple(sorted(set(targets))) != targets:
        raise ValueError("target_counts must be strictly increasing.")
    if targets[-1] > sample_count:
        raise ValueError("largest target exceeds sample count.")
    if not math.isfinite(dedup_threshold):
        raise ValueError("dedup_threshold must be finite.")
    if dedup_threshold >= 0.0 and not 0.0 < dedup_threshold <= 1.0:
        raise ValueError("dedup_threshold must be in (0, 1] or negative to disable.")

    normalized_seed_keys = [normalize_sample_key(str(key)) for key in seed_keys]
    if len(normalized_seed_keys) != len(set(normalized_seed_keys)):
        raise ValueError("seed_keys contain duplicates.")
    key_to_index = {key: index for index, key in enumerate(keys)}
    missing_seeds = sorted(set(normalized_seed_keys) - set(key_to_index))
    if missing_seeds:
        raise ValueError(f"seed_keys are absent from the catalog: {missing_seeds[:3]!r}")
    if len(normalized_seed_keys) > targets[0]:
        raise ValueError("seed count exceeds the smallest target.")

    ranked_by_cluster: dict[int, list[int]] = {}
    for index, cluster_id in enumerate(cluster_tensor.tolist()):
        ranked_by_cluster.setdefault(cluster_id, []).append(index)
    for indices in ranked_by_cluster.values():
        indices.sort(key=lambda index: (-float(score_tensor[index]), keys[index]))

    selected: set[str] = set(normalized_seed_keys)
    selection_order: list[str] = sorted(selected)
    selected_by_cluster: dict[int, list[int]] = {
        cluster_id: [] for cluster_id in ranked_by_cluster
    }
    for key in selected:
        index = key_to_index[key]
        selected_by_cluster[int(cluster_tensor[index])].append(index)

    dedup_skipped_count = {key: 0 for key in keys}
    splits: dict[int, list[str]] = {}
    rounds: list[dict[str, Any]] = []

    def is_duplicate(index: int) -> bool:
        if dedup_threshold < 0.0:
            return False
        cluster_id = int(cluster_tensor[index])
        references = selected_by_cluster[cluster_id]
        if not references:
            return False
        similarities = normalized_features[references] @ normalized_features[index]
        return bool((similarities > dedup_threshold).any())

    def accept(index: int, added: list[str]) -> None:
        key = keys[index]
        if key in selected:
            raise RuntimeError(f"Attempted to select duplicate key {key!r}.")
        selected.add(key)
        selection_order.append(key)
        selected_by_cluster[int(cluster_tensor[index])].append(index)
        added.append(key)

    previous_split: set[str] | None = None
    for target in targets:
        budget = target - len(selected)
        if budget < 0:
            raise RuntimeError("Current nested split already exceeds the next target.")
        remaining_by_cluster = {
            cluster_id: [index for index in ranked if keys[index] not in selected]
            for cluster_id, ranked in ranked_by_cluster.items()
        }
        quotas = allocate_cluster_quotas(
            {cluster_id: len(indices) for cluster_id, indices in remaining_by_cluster.items()},
            budget,
        )
        skipped_before = sum(dedup_skipped_count.values())
        quota_added: list[str] = []
        dedup_backfill_added: list[str] = []
        relaxed_backfill_added: list[str] = []

        for cluster_id in sorted(remaining_by_cluster):
            accepted_for_cluster = 0
            for index in remaining_by_cluster[cluster_id]:
                if accepted_for_cluster >= quotas[cluster_id]:
                    break
                if keys[index] in selected:
                    continue
                if is_duplicate(index):
                    dedup_skipped_count[keys[index]] += 1
                    continue
                accept(index, quota_added)
                accepted_for_cluster += 1

        shortfall = target - len(selected)
        if shortfall:
            global_remaining = sorted(
                (index for index in range(sample_count) if keys[index] not in selected),
                key=lambda index: (-float(score_tensor[index]), keys[index]),
            )
            for index in global_remaining:
                if len(selected) >= target:
                    break
                if is_duplicate(index):
                    dedup_skipped_count[keys[index]] += 1
                    continue
                accept(index, dedup_backfill_added)

        if len(selected) < target:
            global_remaining = sorted(
                (index for index in range(sample_count) if keys[index] not in selected),
                key=lambda index: (-float(score_tensor[index]), keys[index]),
            )
            for index in global_remaining:
                if len(selected) >= target:
                    break
                accept(index, relaxed_backfill_added)

        if len(selected) != target:
            raise RuntimeError(
                f"PC-BACS selected {len(selected)} samples for target {target}."
            )
        current_split = set(selected)
        if previous_split is not None and not previous_split.issubset(current_split):
            raise RuntimeError("Nested split invariant was violated.")
        splits[target] = sorted(current_split)
        rounds.append(
            {
                "target_count": target,
                "budget": budget,
                "quotas": dict(sorted(quotas.items())),
                "quota_selected_count": len(quota_added),
                "dedup_backfill_count": len(dedup_backfill_added),
                "relaxed_backfill_count": len(relaxed_backfill_added),
                "dedup_skips": sum(dedup_skipped_count.values()) - skipped_before,
            }
        )
        previous_split = current_split

    overall_rank = {key: index + 1 for index, key in enumerate(selection_order)}
    return NestedSelectionResult(
        splits=splits,
        selection_order=tuple(selection_order),
        selection_rank={
            target: {key: overall_rank[key] for key in split_keys}
            for target, split_keys in splits.items()
        },
        dedup_skipped_count=dedup_skipped_count,
        rounds=tuple(rounds),
    )


def stable_fingerprint(value: Any) -> str:
    encoded = json.dumps(
        _canonicalize_for_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | os.PathLike) -> str:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Fingerprint source is not a file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_catalog_fingerprint(sample_keys: Sequence[str]) -> str:
    keys = _normalize_unique_keys(sample_keys, name="sample_keys")
    return stable_fingerprint({"version": 1, "sample_keys": sorted(keys)})


def compute_key_order_fingerprint(sample_keys: Sequence[str]) -> str:
    keys = _normalize_unique_keys(sample_keys, name="sample_keys")
    return stable_fingerprint({"version": 1, "sample_keys": keys})


def compute_image_fingerprint(
    sample_keys: Sequence[str],
    image_paths: Sequence[str | os.PathLike] | Mapping[str, str | os.PathLike],
) -> str:
    keys = _normalize_unique_keys(sample_keys, name="sample_keys")
    if isinstance(image_paths, Mapping):
        normalized_paths = {
            normalize_sample_key(str(key)): Path(path) for key, path in image_paths.items()
        }
        if set(normalized_paths) != set(keys):
            missing = sorted(set(keys) - set(normalized_paths))
            extra = sorted(set(normalized_paths) - set(keys))
            raise ValueError(
                f"image path mapping does not match sample keys; missing={missing[:3]!r}, "
                f"extra={extra[:3]!r}."
            )
        pairs = [(key, normalized_paths[key]) for key in keys]
    else:
        paths = [Path(path) for path in image_paths]
        if len(paths) != len(keys):
            raise ValueError("image path count does not match sample key count.")
        pairs = list(zip(keys, paths))

    digest = hashlib.sha256(b"pc-bacs-image-content-v1\0")
    for key, path in sorted(pairs, key=lambda pair: pair[0]):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def preprocessing_fingerprint(input_size: int = 392) -> str:
    return stable_fingerprint(
        {
            "version": PREPROCESSING_VERSION,
            "input_size": int(input_size),
            "resize": "bilinear_antialias",
            "color": "RGB",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        }
    )


def build_feature_cache(
    sample_keys: Sequence[str],
    features: torch.Tensor,
    *,
    catalog_fingerprint: str,
    image_fingerprint: str,
    dino_weight_fingerprint: str,
    preprocessing_fingerprint: str,
    input_size: int = 392,
    feature_type: str = FEATURE_TYPE,
    normalized: bool = False,
    weight_path: str | os.PathLike | None = None,
) -> dict[str, Any]:
    keys = _normalize_unique_keys(sample_keys, name="sample_keys")
    tensor = torch.as_tensor(features).detach().cpu().contiguous()
    if tensor.ndim != 2 or tensor.shape[0] != len(keys):
        raise ValueError("Feature cache tensor must have shape [len(sample_keys), D].")
    if tensor.shape[1] <= 0 or not torch.isfinite(tensor.float()).all():
        raise ValueError("Feature cache tensor has invalid dimension or non-finite values.")
    payload: dict[str, Any] = {
        "format_version": FEATURE_CACHE_FORMAT_VERSION,
        "feature_type": str(feature_type),
        "input_size": int(input_size),
        "sample_keys": keys,
        "key_order_fingerprint": compute_key_order_fingerprint(keys),
        "features": tensor,
        "normalized": bool(normalized),
        "catalog_fingerprint": str(catalog_fingerprint),
        "image_fingerprint": str(image_fingerprint),
        "dino_weight_fingerprint": str(dino_weight_fingerprint),
        "preprocessing_fingerprint": str(preprocessing_fingerprint),
    }
    if weight_path is not None:
        payload["weight_path"] = str(Path(weight_path))
    return payload


def validate_feature_cache(
    cache: Mapping[str, Any],
    *,
    expected_sample_keys: Sequence[str],
    expected_catalog_fingerprint: str,
    expected_image_fingerprint: str,
    expected_dino_weight_fingerprint: str,
    expected_preprocessing_fingerprint: str,
    feature_dim: int = 768,
    input_size: int = 392,
) -> tuple[list[str], torch.Tensor]:
    _expect_cache_field(cache, "format_version", FEATURE_CACHE_FORMAT_VERSION)
    _expect_cache_field(cache, "feature_type", FEATURE_TYPE)
    _expect_cache_field(cache, "input_size", int(input_size))
    _expect_cache_field(cache, "catalog_fingerprint", expected_catalog_fingerprint)
    _expect_cache_field(cache, "image_fingerprint", expected_image_fingerprint)
    _expect_cache_field(cache, "dino_weight_fingerprint", expected_dino_weight_fingerprint)
    _expect_cache_field(
        cache, "preprocessing_fingerprint", expected_preprocessing_fingerprint
    )

    expected_keys = _normalize_unique_keys(expected_sample_keys, name="expected_sample_keys")
    keys = _normalize_unique_keys(_require_cache_field(cache, "sample_keys"), name="cache sample_keys")
    if keys != expected_keys:
        raise ValueError("Feature cache sample key order does not match the current catalog.")
    _expect_cache_field(cache, "key_order_fingerprint", compute_key_order_fingerprint(keys))

    features = torch.as_tensor(_require_cache_field(cache, "features")).detach().cpu()
    if features.ndim != 2 or tuple(features.shape) != (len(keys), int(feature_dim)):
        raise ValueError(
            f"Feature cache shape mismatch: expected {(len(keys), int(feature_dim))}, "
            f"got {tuple(features.shape)}."
        )
    if not torch.isfinite(features.float()).all():
        raise ValueError("Feature cache contains NaN or infinity.")
    normalized = _require_cache_field(cache, "normalized")
    if not isinstance(normalized, bool):
        raise ValueError("Feature cache normalized flag must be boolean.")
    return keys, features


def build_score_cache(
    sample_keys: Sequence[str],
    boundary_disagreement: Sequence[float] | torch.Tensor,
    global_disagreement: Sequence[float] | torch.Tensor,
    scores: Sequence[float] | torch.Tensor,
    *,
    selector_fingerprint: str,
    catalog_fingerprint: str,
    image_fingerprint: str,
    preprocessing_fingerprint: str,
    score_formula_version: str = SCORE_FORMULA_VERSION,
) -> dict[str, Any]:
    keys = _normalize_unique_keys(sample_keys, name="sample_keys")
    boundary = _validated_score_vector(boundary_disagreement, len(keys), "boundary_disagreement")
    global_value = _validated_score_vector(global_disagreement, len(keys), "global_disagreement")
    score_value = _validated_score_vector(scores, len(keys), "scores")
    if not torch.allclose(
        score_value,
        boundary * (1.0 - global_value),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("Score cache values do not match D_bd * (1 - D_all).")
    return {
        "format_version": SCORE_CACHE_FORMAT_VERSION,
        "selector_fingerprint": str(selector_fingerprint),
        "catalog_fingerprint": str(catalog_fingerprint),
        "image_fingerprint": str(image_fingerprint),
        "preprocessing_fingerprint": str(preprocessing_fingerprint),
        "score_formula_version": str(score_formula_version),
        "sample_keys": keys,
        "key_order_fingerprint": compute_key_order_fingerprint(keys),
        "boundary_disagreement": boundary,
        "global_disagreement": global_value,
        "scores": score_value,
    }


def validate_score_cache(
    cache: Mapping[str, Any],
    *,
    expected_sample_keys: Sequence[str],
    expected_selector_fingerprint: str,
    expected_catalog_fingerprint: str,
    expected_image_fingerprint: str,
    expected_preprocessing_fingerprint: str,
    expected_score_formula_version: str = SCORE_FORMULA_VERSION,
) -> dict[str, torch.Tensor]:
    _expect_cache_field(cache, "format_version", SCORE_CACHE_FORMAT_VERSION)
    _expect_cache_field(cache, "selector_fingerprint", expected_selector_fingerprint)
    _expect_cache_field(cache, "catalog_fingerprint", expected_catalog_fingerprint)
    _expect_cache_field(cache, "image_fingerprint", expected_image_fingerprint)
    _expect_cache_field(
        cache, "preprocessing_fingerprint", expected_preprocessing_fingerprint
    )
    _expect_cache_field(
        cache, "score_formula_version", expected_score_formula_version
    )
    expected_keys = _normalize_unique_keys(expected_sample_keys, name="expected_sample_keys")
    keys = _normalize_unique_keys(_require_cache_field(cache, "sample_keys"), name="cache sample_keys")
    if keys != expected_keys:
        raise ValueError("Score cache sample key order does not match the current catalog.")
    _expect_cache_field(cache, "key_order_fingerprint", compute_key_order_fingerprint(keys))

    boundary = _validated_score_vector(
        _require_cache_field(cache, "boundary_disagreement"), len(keys), "boundary_disagreement"
    )
    global_value = _validated_score_vector(
        _require_cache_field(cache, "global_disagreement"), len(keys), "global_disagreement"
    )
    score_value = _validated_score_vector(
        _require_cache_field(cache, "scores"), len(keys), "scores"
    )
    if not torch.allclose(
        score_value,
        boundary * (1.0 - global_value),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("Score cache values do not match D_bd * (1 - D_all).")
    return {
        "boundary_disagreement": boundary,
        "global_disagreement": global_value,
        "scores": score_value,
    }


def atomic_torch_save(
    payload: Any,
    path: str | os.PathLike,
    *,
    refuse_mismatch: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = _torch_load(path)
        if _payload_equal(existing, payload):
            return
        if refuse_mismatch:
            raise FileExistsError(f"Refusing to overwrite different artifact: {path}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_json_save(
    payload: Mapping[str, Any],
    path: str | os.PathLike,
    *,
    refuse_mismatch: bool = True,
) -> None:
    canonical = _canonicalize_for_json(payload)
    serialized = json.dumps(
        canonical,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Existing JSON artifact is invalid: {path}") from error
        if existing == canonical:
            return
        if refuse_mismatch:
            raise FileExistsError(f"Refusing to overwrite different artifact: {path}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent), text=True
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(serialized, encoding="utf-8")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def save_split_keys(path: str | os.PathLike, sample_keys: Sequence[str]) -> str:
    normalized = [normalize_sample_key(str(key)) for key in sample_keys]
    if len(normalized) != len(set(normalized)):
        raise ValueError("split contains duplicate sample keys.")
    if any(not key for key in normalized):
        raise ValueError("split contains an empty sample key.")
    normalized.sort()
    atomic_torch_save(normalized, path, refuse_mismatch=True)
    return compute_labeled_split_fingerprint(normalized)


def load_split_keys(
    path: str | os.PathLike,
    *,
    catalog_keys: Sequence[str] | None = None,
) -> list[str]:
    payload = _torch_load(path)
    if not isinstance(payload, list) or not all(isinstance(key, str) for key in payload):
        raise ValueError("PC-BACS split must contain a plain list[str].")
    keys = _normalize_unique_keys(payload, name="split")
    if keys != sorted(keys):
        raise ValueError("PC-BACS split keys must be sorted.")
    if catalog_keys is not None:
        catalog = set(_normalize_unique_keys(catalog_keys, name="catalog_keys"))
        missing = sorted(set(keys) - catalog)
        if missing:
            raise ValueError(f"Split keys are absent from the catalog: {missing[:3]!r}")
    return keys


def build_runtime_metadata() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn_version,
        "cuda": torch.version.cuda,
    }


def build_selection_manifest(
    *,
    config: PCBACSConfig | Mapping[str, Any],
    dataset: Mapping[str, Any],
    selector: Mapping[str, Any],
    outputs: Mapping[str | int, Any],
    repo_commit: str | None = None,
    selection_result: NestedSelectionResult | None = None,
    runtime: Mapping[str, Any] | None = None,
    fingerprints: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(config, PCBACSConfig):
        config.validate()
        config_dict = asdict(config)
    elif isinstance(config, Mapping):
        config_dict = dict(config)
    else:
        raise TypeError("config must be PCBACSConfig or a mapping.")

    selection: dict[str, Any] = {
        "clusters": int(config_dict["n_clusters"]),
        "target_counts": list(config_dict["target_counts"]),
        "selector_seed_count": int(config_dict.get("selector_seed_count", 40)),
        "dedup_threshold": float(config_dict["dedup_threshold"]),
        "seed": int(config_dict["random_seed"]),
        "score_formula": "D_bd * (1 - D_all)",
        "score_formula_version": str(
            config_dict.get("score_formula_version", SCORE_FORMULA_VERSION)
        ),
    }
    if selection_result is not None:
        selection["rounds"] = list(selection_result.rounds)
        selection["selected_counts"] = {
            str(target): len(keys) for target, keys in selection_result.splits.items()
        }

    manifest: dict[str, Any] = {
        "format_version": MANIFEST_FORMAT_VERSION,
        "method": "PC-BACS",
        "repo_commit": repo_commit,
        "config": config_dict,
        "dataset": dict(dataset),
        "selector": dict(selector),
        "selection": selection,
        "outputs": {str(key): value for key, value in outputs.items()},
        "runtime": dict(runtime) if runtime is not None else build_runtime_metadata(),
        "fingerprints": dict(fingerprints or {}),
    }
    if extra:
        manifest["extra"] = dict(extra)
    return _canonicalize_for_json(manifest)


def _normalize_unique_keys(sample_keys: Sequence[str], *, name: str) -> list[str]:
    if isinstance(sample_keys, (str, bytes)) or not isinstance(sample_keys, Sequence):
        raise TypeError(f"{name} must be a sequence of strings.")
    keys: list[str] = []
    for value in sample_keys:
        if not isinstance(value, str):
            raise TypeError(f"{name} must contain only strings.")
        key = normalize_sample_key(value)
        if not key:
            raise ValueError(f"{name} contains an empty key.")
        keys.append(key)
    if len(keys) != len(set(keys)):
        raise ValueError(f"{name} contains duplicate keys.")
    return keys


def _validated_score_vector(value: Any, length: int, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).detach().cpu().contiguous()
    if tensor.ndim != 1 or tensor.numel() != length:
        raise ValueError(f"{name} must have shape [{length}].")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN or infinity.")
    if bool(((tensor < 0.0) | (tensor > 1.0)).any()):
        raise ValueError(f"{name} must be in [0, 1].")
    return tensor


def _require_cache_field(cache: Mapping[str, Any], name: str) -> Any:
    if not isinstance(cache, Mapping):
        raise TypeError("cache must be a mapping.")
    if name not in cache:
        raise ValueError(f"Cache is missing required field {name!r}.")
    return cache[name]


def _expect_cache_field(cache: Mapping[str, Any], name: str, expected: Any) -> None:
    actual = _require_cache_field(cache, name)
    if actual != expected:
        raise ValueError(
            f"Cache field {name!r} mismatch: expected {expected!r}, got {actual!r}."
        )


def _canonicalize_for_json(value: Any) -> Any:
    if is_dataclass(value):
        return _canonicalize_for_json(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize_for_json(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_for_json(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _canonicalize_for_json(value.item())
    if isinstance(value, torch.Tensor):
        return _canonicalize_for_json(value.detach().cpu().tolist())
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("JSON metadata contains NaN or infinity.")
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"Unsupported metadata value type: {type(value).__name__}")


def _torch_load(path: str | os.PathLike) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _payload_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and tuple(left.shape) == tuple(right.shape) and torch.equal(left, right)
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _payload_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _payload_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    try:
        return bool(left == right)
    except (TypeError, ValueError, RuntimeError):
        return False


__all__ = [
    "FEATURE_CACHE_FORMAT_VERSION",
    "FEATURE_TYPE",
    "KMeansResult",
    "MANIFEST_FORMAT_VERSION",
    "NestedSelectionResult",
    "PREPROCESSING_VERSION",
    "SCORE_CACHE_FORMAT_VERSION",
    "SCORE_FORMULA_VERSION",
    "allocate_cluster_quotas",
    "atomic_json_save",
    "atomic_torch_save",
    "build_feature_cache",
    "build_nested_splits",
    "build_runtime_metadata",
    "build_score_cache",
    "build_selection_manifest",
    "compute_catalog_fingerprint",
    "compute_image_fingerprint",
    "compute_key_order_fingerprint",
    "compute_pc_bacs_score",
    "file_sha256",
    "fit_dino_kmeans",
    "load_split_keys",
    "normalize_dino_features",
    "preprocessing_fingerprint",
    "save_split_keys",
    "score_pool",
    "sobel_magnitude",
    "stable_fingerprint",
    "validate_feature_cache",
    "validate_score_cache",
]
