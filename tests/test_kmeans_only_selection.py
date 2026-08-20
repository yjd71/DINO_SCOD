from __future__ import annotations

import numpy as np
import pytest
import torch

import utils.kmeans_only as module
from utils.kmeans_only import (
    KMEANS_ONLY_PROTOCOL_VERSION,
    allocate_sqrt_quotas,
    build_kmeans_only_nested_splits,
    fit_kmeans_only,
    labeled_names_from_keys,
    normalize_kmeans_features,
)


def test_fit_kmeans_uses_frozen_parameters_and_sample_key_seed_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeKMeans:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def fit(self, matrix: np.ndarray) -> "FakeKMeans":
            captured["matrix"] = matrix.copy()
            self.labels_ = np.asarray([0, 0, 1, 1], dtype=np.int64)
            return self

    monkeypatch.setattr(module, "KMeans", FakeKMeans)
    result = fit_kmeans_only(
        ["SET/b", "SET/a", "SET/d", "SET/c"],
        torch.tensor(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            dtype=torch.float32,
        ),
        n_clusters=2,
        random_seed=2025,
    )

    assert captured["n_clusters"] == 2
    assert captured["random_state"] == 2025
    assert captured["n_init"] == 10
    assert captured["algorithm"] == "lloyd"
    assert result.seed_keys == ("SET/a", "SET/c")
    assert result.cluster_ids.tolist() == [0, 0, 1, 1]
    assert result.center_distances.tolist() == pytest.approx([0.0] * 4)
    assert result.centers.dtype == torch.float32


@pytest.mark.parametrize(
    ("capacities", "budget", "expected"),
    [
        ({0: 9, 1: 4, 2: 1}, 0, {0: 0, 1: 0, 2: 0}),
        ({0: 9, 1: 4, 2: 1}, 7, {0: 4, 1: 2, 2: 1}),
        ({0: 1, 1: 1, 2: 1}, 2, {0: 1, 1: 1, 2: 0}),
    ],
)
def test_sqrt_quota_is_exact_and_stable(
    capacities: dict[int, int],
    budget: int,
    expected: dict[int, int],
) -> None:
    result = allocate_sqrt_quotas(capacities, budget)
    assert result == expected
    assert sum(result.values()) == budget
    assert all(result[key] <= capacities[key] for key in result)


def test_sqrt_quota_rejects_capacity_overflow() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        allocate_sqrt_quotas({0: 1, 1: 0}, 2)


def test_nested_selection_is_exact_stable_and_center_distance_only() -> None:
    keys = [
        "SET/a0",
        "SET/a1",
        "SET/a2",
        "SET/a3",
        "SET/b0",
        "SET/b1",
        "SET/b2",
        "SET/b3",
    ]
    result = build_kmeans_only_nested_splits(
        keys,
        cluster_ids=[0, 0, 0, 0, 1, 1, 1, 1],
        center_distances=[0.0, 0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.3],
        seed_keys=["SET/a0", "SET/b0"],
        target_counts=(3, 5, 8),
    )

    assert result.protocol_version == KMEANS_ONLY_PROTOCOL_VERSION
    assert result.seed_keys == ("SET/a0", "SET/b0")
    assert set(result.splits[3]) == {"SET/a0", "SET/b0", "SET/a1"}
    assert set(result.splits[3]) < set(result.splits[5]) < set(result.splits[8])
    assert [len(result.splits[target]) for target in (3, 5, 8)] == [3, 5, 8]
    assert [round_record.budget for round_record in result.rounds] == [1, 2, 3]
    assert result.rounds[0].quotas == {0: 1, 1: 0}
    assert len(result.selection_order) == len(set(result.selection_order)) == 8


def test_center_distance_tie_uses_sample_key() -> None:
    result = build_kmeans_only_nested_splits(
        ["SET/seed0", "SET/z", "SET/a", "SET/seed1"],
        cluster_ids=[0, 0, 0, 1],
        center_distances=[0.0, 0.5, 0.5, 0.0],
        seed_keys=["SET/seed0", "SET/seed1"],
        target_counts=(3,),
    )
    assert "SET/a" in result.splits[3]
    assert "SET/z" not in result.splits[3]


def test_seed_must_be_center_nearest_and_cover_each_cluster() -> None:
    with pytest.raises(ValueError, match="not the center-nearest"):
        build_kmeans_only_nested_splits(
            ["SET/a", "SET/b", "SET/c"],
            cluster_ids=[0, 0, 1],
            center_distances=[0.0, 0.2, 0.0],
            seed_keys=["SET/b", "SET/c"],
            target_counts=(2, 3),
        )
    with pytest.raises(ValueError, match="cover every cluster"):
        build_kmeans_only_nested_splits(
            ["SET/a", "SET/b", "SET/c"],
            cluster_ids=[0, 0, 1],
            center_distances=[0.0, 0.2, 0.0],
            seed_keys=["SET/a", "SET/b"],
            target_counts=(2, 3),
        )


@pytest.mark.parametrize(
    "features",
    [
        torch.ones(3),
        torch.tensor([[1.0, float("nan")]]),
        torch.zeros(2, 3),
    ],
)
def test_feature_validation_rejects_bad_shape_nan_and_zero_rows(
    features: torch.Tensor,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalize_kmeans_features(features)


def test_normalized_features_are_float32_unit_rows() -> None:
    normalized = normalize_kmeans_features(
        torch.tensor([[3.0, 4.0], [0.0, 2.0]], dtype=torch.float64),
        expected_length=2,
    )
    assert normalized.dtype == torch.float32
    assert torch.linalg.vector_norm(normalized, dim=1).tolist() == pytest.approx(
        [1.0, 1.0]
    )


def test_txt_names_preserve_dots_and_reject_cross_dataset_duplicates() -> None:
    assert labeled_names_from_keys(["A/animal.v2", "B/unique"]) == [
        "animal.v2",
        "unique",
    ]
    with pytest.raises(ValueError, match="duplicate labeled stems"):
        labeled_names_from_keys(["A/shared", "B/shared"])
