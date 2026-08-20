from __future__ import annotations

import inspect

import pytest
import torch

from utils.kmeans_dbd import (
    KMEANS_DBD_DEDUP_PROTOCOL_VERSION,
    build_kmeans_dbd_nested_splits,
)


def test_boundary_ranking_is_exact_nested_stable_and_ignores_center_distance() -> None:
    keys = [f"SET/sample_{index}" for index in range(8)]
    features = torch.eye(8)
    kwargs = {
        "sample_keys": keys,
        "features": features,
        "cluster_ids": [0, 0, 0, 0, 1, 1, 1, 1],
        "center_distances": [0.0, 0.9, 0.1, 0.2, 0.0, 0.9, 0.1, 0.2],
        "boundary_disagreement": [0.0, 0.9, 0.8, 0.7, 0.0, 0.6, 0.5, 0.4],
        "seed_keys": ["SET/sample_0", "SET/sample_4"],
        "target_counts": (3, 5, 8),
        "dedup_threshold": 0.95,
    }

    first = build_kmeans_dbd_nested_splits(**kwargs)
    second = build_kmeans_dbd_nested_splits(**kwargs)

    assert first.protocol_version == KMEANS_DBD_DEDUP_PROTOCOL_VERSION
    assert first.splits == second.splits
    assert first.selection_order == second.selection_order
    assert [len(first.splits[target]) for target in (3, 5, 8)] == [3, 5, 8]
    assert set(first.splits[3]) < set(first.splits[5]) < set(first.splits[8])
    assert "SET/sample_1" in first.splits[3]
    assert "SET/sample_2" not in first.splits[3]
    assert [record.budget for record in first.rounds] == [1, 2, 3]


def test_boundary_ties_use_sample_key() -> None:
    result = build_kmeans_dbd_nested_splits(
        ["SET/seed", "SET/z", "SET/a"],
        torch.eye(3),
        cluster_ids=[0, 0, 0],
        center_distances=[0.0, 0.5, 0.5],
        boundary_disagreement=[0.0, 0.8, 0.8],
        seed_keys=["SET/seed"],
        target_counts=(2,),
        dedup_threshold=0.95,
    )
    assert result.splits[2] == ["SET/a", "SET/seed"]


def test_dedup_uses_seed_and_current_round_but_never_crosses_clusters() -> None:
    keys = [
        "SET/seed0",
        "SET/c0_duplicate",
        "SET/c0_unique",
        "SET/c0_other",
        "SET/seed1",
        "SET/c1_cross_cluster_match",
        "SET/c1_unique",
        "SET/c1_other",
    ]
    features = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    result = build_kmeans_dbd_nested_splits(
        keys,
        features,
        cluster_ids=[0, 0, 0, 0, 1, 1, 1, 1],
        center_distances=[0.0, 0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.3],
        boundary_disagreement=[0.0, 0.99, 0.9, 0.8, 0.0, 0.99, 0.9, 0.8],
        seed_keys=["SET/seed0", "SET/seed1"],
        target_counts=(4,),
        dedup_threshold=0.95,
    )

    assert "SET/c0_duplicate" not in result.splits[4]
    assert "SET/c0_unique" in result.splits[4]
    assert "SET/c1_cross_cluster_match" in result.splits[4]
    assert result.decisions["SET/c0_duplicate"].skip_count == 1
    assert result.decisions["SET/c0_duplicate"].reference_key == "SET/seed0"
    assert result.rounds[0].dedup_skips == 1


def test_similarity_equal_to_threshold_is_allowed() -> None:
    result = build_kmeans_dbd_nested_splits(
        ["SET/seed", "SET/same"],
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        cluster_ids=[0, 0],
        center_distances=[0.0, 0.1],
        boundary_disagreement=[0.0, 1.0],
        seed_keys=["SET/seed"],
        target_counts=(2,),
        dedup_threshold=1.0,
    )
    assert result.splits[2] == ["SET/same", "SET/seed"]
    assert result.rounds[0].dedup_skips == 0


def test_global_keep_dedup_and_relaxed_backfill_make_count_exact() -> None:
    keys = [
        "SET/seed0",
        "SET/c0_a",
        "SET/c0_b",
        "SET/c0_c",
        "SET/c0_d",
        "SET/seed1",
        "SET/c1_a",
        "SET/c1_b",
        "SET/c1_c",
        "SET/c1_d",
    ]
    features = torch.zeros(10, 8)
    features[:5, 0] = 1.0
    for offset, index in enumerate(range(5, 10), start=1):
        features[index, offset] = 1.0
    result = build_kmeans_dbd_nested_splits(
        keys,
        features,
        cluster_ids=[0] * 5 + [1] * 5,
        center_distances=[0.0, 0.1, 0.2, 0.3, 0.4] * 2,
        boundary_disagreement=[0.0, 0.99, 0.98, 0.97, 0.96, 0.0, 0.5, 0.4, 0.3, 0.2],
        seed_keys=["SET/seed0", "SET/seed1"],
        target_counts=(9,),
        dedup_threshold=0.95,
    )

    record = result.rounds[0]
    assert len(result.splits[9]) == 9
    assert record.quota_selected_count == 3
    assert record.dedup_backfill_count == 1
    assert record.relaxed_backfill_count == 3
    assert record.dedup_skips > 0
    assert sum(decision.relaxed_backfill for decision in result.decisions.values()) == 3


def test_seed_must_be_center_nearest_and_cover_every_cluster() -> None:
    with pytest.raises(ValueError, match="not the center-nearest"):
        build_kmeans_dbd_nested_splits(
            ["SET/a", "SET/b", "SET/c"],
            torch.eye(3),
            cluster_ids=[0, 0, 1],
            center_distances=[0.0, 0.2, 0.0],
            boundary_disagreement=[0.1, 0.2, 0.3],
            seed_keys=["SET/b", "SET/c"],
            target_counts=(3,),
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("features", torch.tensor([[1.0, 0.0], [float("nan"), 1.0]]), "NaN"),
        ("cluster_ids", [0.0, 0.0], "integers"),
        ("center_distances", [0.0, -0.1], "below"),
        ("boundary_disagreement", [0.0, float("nan")], "NaN"),
        ("boundary_disagreement", [0.0, 1.1], "above"),
    ],
)
def test_invalid_aligned_inputs_are_rejected(field: str, value, match: str) -> None:
    kwargs = {
        "sample_keys": ["SET/seed", "SET/item"],
        "features": torch.eye(2),
        "cluster_ids": [0, 0],
        "center_distances": [0.0, 0.1],
        "boundary_disagreement": [0.0, 0.5],
        "seed_keys": ["SET/seed"],
        "target_counts": (2,),
        "dedup_threshold": 0.95,
    }
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError), match=match):
        build_kmeans_dbd_nested_splits(**kwargs)


def test_algorithm_api_cannot_receive_global_disagreement_or_pc_bacs_value() -> None:
    parameters = inspect.signature(build_kmeans_dbd_nested_splits).parameters
    assert "boundary_disagreement" in parameters
    assert "global_disagreement" not in parameters
    assert "scores" not in parameters
