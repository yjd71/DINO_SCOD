"""Deterministic CPU-float32 greedy acquisition for BPUS-v2."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .formulas import SOFT_REWARD, compute_bpus_v2_novelty, compute_utility
from .prototype import PROTOTYPE_DIM


@dataclass(frozen=True)
class AcquisitionBPUSV2Result:
    """Nested splits and diagnostics captured at each acquisition step."""

    splits: dict[int, list[str]]
    acquired_keys: list[str]
    utility: torch.Tensor
    value: torch.Tensor
    novelty: torch.Tensor
    max_similarity: torch.Tensor


def _unique_keys(values: Sequence[str], name: str) -> list[str]:
    keys = [str(value) for value in values]
    if any(not key for key in keys):
        raise ValueError(f"{name} cannot contain empty sample keys.")
    if len(keys) != len(set(keys)):
        raise ValueError(f"{name} must contain unique sample keys.")
    return keys


def _target_counts(values: Sequence[int]) -> tuple[int, ...]:
    counts = tuple(int(value) for value in values)
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("target_counts must contain positive integers.")
    if any(left >= right for left, right in zip(counts, counts[1:])):
        raise ValueError("target_counts must be strictly increasing.")
    return counts


def _value_vector(values, length: int) -> torch.Tensor:
    result = torch.as_tensor(values, dtype=torch.float32, device="cpu").reshape(-1)
    if result.numel() != length:
        raise ValueError("values must have one entry per sample key.")
    if not torch.isfinite(result).all():
        raise ValueError("values must be finite.")
    if bool(((result < 0.0) | (result > 1.0)).any()):
        raise ValueError("values must lie in [0,1].")
    return result.contiguous()


def greedy_acquire_bpus_v2(
    sample_keys: Sequence[str],
    prototypes: torch.Tensor,
    values,
    valid_boundary,
    bootstrap_keys: Sequence[str],
    target_counts: Sequence[int],
    *,
    reward_mode: str = SOFT_REWARD,
) -> AcquisitionBPUSV2Result:
    """Greedily maximize value with the selected novelty reward.

    Every similarity, incremental update, and tie comparison is performed on
    detached CPU float32 tensors.  Ties resolve by higher utility, value,
    novelty, and finally the lexically smaller sample key.
    """

    keys = _unique_keys(sample_keys, "sample_keys")
    bootstrap = _unique_keys(bootstrap_keys, "bootstrap_keys")
    counts = _target_counts(target_counts)
    sample_count = len(keys)
    if counts[0] != len(bootstrap):
        raise ValueError("The first target count must equal the bootstrap size.")
    if counts[-1] > sample_count:
        raise ValueError("The largest target count exceeds the catalog size.")
    index_by_key = {key: index for index, key in enumerate(keys)}
    missing = sorted(set(bootstrap) - set(index_by_key))
    if missing:
        raise ValueError(f"bootstrap_keys are absent from the catalog: {missing[:3]}")

    if not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 2:
        raise ValueError("prototypes must have shape [N,128].")
    if tuple(prototypes.shape) != (sample_count, PROTOTYPE_DIM):
        raise ValueError(
            f"prototypes must have shape [{sample_count},{PROTOTYPE_DIM}], "
            f"found {tuple(prototypes.shape)}."
        )
    features = prototypes.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not torch.isfinite(features).all():
        raise ValueError("prototypes must be finite.")
    norms = torch.linalg.vector_norm(features, dim=1, keepdim=True)
    features = torch.where(
        norms > 0.0,
        features / norms.clamp_min(1e-12),
        torch.zeros_like(features),
    )

    value_vector = _value_vector(values, sample_count)
    valid = torch.as_tensor(valid_boundary, dtype=torch.bool, device="cpu").reshape(-1)
    if valid.numel() != sample_count:
        raise ValueError("valid_boundary must have one entry per sample key.")
    features = features.clone()
    features[~valid] = 0.0

    bootstrap_set = set(bootstrap)
    remaining_indices = [
        index
        for index, key in enumerate(keys)
        if key not in bootstrap_set and bool(valid[index])
    ]
    required = counts[-1] - len(bootstrap)
    if len(remaining_indices) < required:
        raise RuntimeError(
            "Insufficient valid-boundary candidates: "
            f"need {required}, found {len(remaining_indices)}."
        )

    bootstrap_indices = [index_by_key[key] for key in bootstrap]
    if remaining_indices and bootstrap_indices:
        initial = features[remaining_indices] @ features[bootstrap_indices].T
        current_similarity = initial.max(dim=1).values
    else:
        current_similarity = torch.zeros(len(remaining_indices), dtype=torch.float32)

    acquired_keys: list[str] = []
    acquired_utility: list[float] = []
    acquired_value: list[float] = []
    acquired_novelty: list[float] = []
    acquired_similarity: list[float] = []

    for _ in range(required):
        clipped_similarity = current_similarity.clamp(0.0, 1.0)
        novelty = compute_bpus_v2_novelty(current_similarity)
        candidate_values = value_vector[remaining_indices]
        utility = compute_utility(candidate_values, novelty, mode=reward_mode)
        best_position = min(
            range(len(remaining_indices)),
            key=lambda position: (
                -float(utility[position]),
                -float(candidate_values[position]),
                -float(novelty[position]),
                keys[remaining_indices[position]],
            ),
        )
        best_index = remaining_indices[best_position]
        acquired_keys.append(keys[best_index])
        acquired_utility.append(float(utility[best_position]))
        acquired_value.append(float(candidate_values[best_position]))
        acquired_novelty.append(float(novelty[best_position]))
        acquired_similarity.append(float(clipped_similarity[best_position]))

        del remaining_indices[best_position]
        keep = torch.ones(current_similarity.shape[0], dtype=torch.bool)
        keep[best_position] = False
        current_similarity = current_similarity[keep]
        if remaining_indices:
            similarity_to_selected = features[remaining_indices] @ features[best_index]
            current_similarity = torch.maximum(
                current_similarity, similarity_to_selected
            )

    splits = {
        count: sorted(bootstrap + acquired_keys[: count - len(bootstrap)])
        for count in counts
    }
    return AcquisitionBPUSV2Result(
        splits=splits,
        acquired_keys=acquired_keys,
        utility=torch.tensor(acquired_utility, dtype=torch.float32),
        value=torch.tensor(acquired_value, dtype=torch.float32),
        novelty=torch.tensor(acquired_novelty, dtype=torch.float32),
        max_similarity=torch.tensor(acquired_similarity, dtype=torch.float32),
    )


AcquisitionResult = AcquisitionBPUSV2Result
greedy_acquire = greedy_acquire_bpus_v2
