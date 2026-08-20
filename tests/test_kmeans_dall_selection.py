from __future__ import annotations

import inspect

import pytest
import torch

from utils.kmeans_dall import (
    KMEANS_DALL_DEDUP_PROTOCOL_VERSION,
    build_kmeans_dall_nested_splits,
)


def _base_kwargs() -> dict[str, object]:
    return {
        "sample_keys": [f"SET/sample_{index}" for index in range(8)],
        "features": torch.eye(8),
        "cluster_ids": [0, 0, 0, 0, 1, 1, 1, 1],
        "center_distances": [0.0, 0.9, 0.1, 0.2, 0.0, 0.9, 0.1, 0.2],
        "global_disagreement": [0.0, 0.9, 0.8, 0.7, 0.0, 0.6, 0.5, 0.4],
        "seed_keys": ["SET/sample_0", "SET/sample_4"],
        "target_counts": (3, 5, 8),
        "dedup_threshold": 0.95,
    }


def test_high_and_low_directions_are_exact_nested_and_deterministic() -> None:
    high = build_kmeans_dall_nested_splits(**_base_kwargs(), direction="high")
    low = build_kmeans_dall_nested_splits(**_base_kwargs(), direction="low")
    high_again = build_kmeans_dall_nested_splits(**_base_kwargs(), direction="high")

    assert high.protocol_version == KMEANS_DALL_DEDUP_PROTOCOL_VERSION
    assert high.direction == "high"
    assert low.direction == "low"
    assert high.splits == high_again.splits
    assert high.selection_order == high_again.selection_order
    for result in (high, low):
        assert [len(result.splits[target]) for target in (3, 5, 8)] == [3, 5, 8]
        assert set(result.splits[3]) < set(result.splits[5]) < set(result.splits[8])
        assert [record.budget for record in result.rounds] == [1, 2, 3]
    assert "SET/sample_1" in high.splits[3]
    assert "SET/sample_3" in low.splits[3]


def test_direction_ties_use_sample_key() -> None:
    kwargs = {
        "sample_keys": ["SET/seed", "SET/z", "SET/a"],
        "features": torch.eye(3),
        "cluster_ids": [0, 0, 0],
        "center_distances": [0.0, 0.5, 0.5],
        "global_disagreement": [0.5, 0.8, 0.8],
        "seed_keys": ["SET/seed"],
        "target_counts": (2,),
        "dedup_threshold": 0.95,
    }
    for direction in ("high", "low"):
        result = build_kmeans_dall_nested_splits(**kwargs, direction=direction)
        assert result.splits[2] == ["SET/a", "SET/seed"]


def test_low_direction_uses_original_float32_values_without_complement() -> None:
    smaller = torch.tensor(0.05, dtype=torch.float32)
    larger = torch.nextafter(smaller, torch.tensor(1.0, dtype=torch.float32))
    assert smaller < larger
    assert (torch.tensor(1.0) - smaller) == (torch.tensor(1.0) - larger)

    result = build_kmeans_dall_nested_splits(
        ["SET/seed", "SET/z", "SET/a"],
        torch.eye(3),
        cluster_ids=[0, 0, 0],
        center_distances=[0.0, 0.5, 0.5],
        global_disagreement=torch.stack((torch.tensor(0.5), smaller, larger)),
        seed_keys=["SET/seed"],
        direction="low",
        target_counts=(2,),
        dedup_threshold=0.95,
    )
    assert result.splits[2] == ["SET/seed", "SET/z"]


def test_dedup_is_same_cluster_only_and_uses_seed_reference() -> None:
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
    result = build_kmeans_dall_nested_splits(
        keys,
        features,
        cluster_ids=[0, 0, 0, 0, 1, 1, 1, 1],
        center_distances=[0.0, 0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.3],
        global_disagreement=[0.0, 0.99, 0.9, 0.8, 0.0, 0.99, 0.9, 0.8],
        seed_keys=["SET/seed0", "SET/seed1"],
        direction="high",
        target_counts=(4,),
        dedup_threshold=0.95,
    )

    assert "SET/c0_duplicate" not in result.splits[4]
    assert "SET/c0_unique" in result.splits[4]
    assert "SET/c1_cross_cluster_match" in result.splits[4]
    assert result.decisions["SET/c0_duplicate"].skip_count == 1
    assert result.decisions["SET/c0_duplicate"].reference_key == "SET/seed0"


def test_similarity_equal_to_threshold_is_allowed() -> None:
    result = build_kmeans_dall_nested_splits(
        ["SET/seed", "SET/same"],
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        cluster_ids=[0, 0],
        center_distances=[0.0, 0.1],
        global_disagreement=[0.0, 1.0],
        seed_keys=["SET/seed"],
        direction="high",
        target_counts=(2,),
        dedup_threshold=1.0,
    )
    assert result.splits[2] == ["SET/same", "SET/seed"]
    assert result.rounds[0].dedup_skips == 0


def test_keep_dedup_and_relaxed_backfill_make_count_exact() -> None:
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
    result = build_kmeans_dall_nested_splits(
        keys,
        features,
        cluster_ids=[0] * 5 + [1] * 5,
        center_distances=[0.0, 0.1, 0.2, 0.3, 0.4] * 2,
        global_disagreement=[0.0, 0.99, 0.98, 0.97, 0.96, 0.0, 0.5, 0.4, 0.3, 0.2],
        seed_keys=["SET/seed0", "SET/seed1"],
        direction="high",
        target_counts=(9,),
        dedup_threshold=0.95,
    )

    record = result.rounds[0]
    assert len(result.splits[9]) == 9
    assert record.quota_selected_count == 3
    assert record.dedup_backfill_count == 1
    assert record.relaxed_backfill_count == 3
    assert record.dedup_skips > 0


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("features", torch.tensor([[1.0, 0.0], [float("nan"), 1.0]]), "NaN"),
        ("cluster_ids", [0.0, 0.0], "integers"),
        ("center_distances", [0.0, -0.1], "below"),
        ("global_disagreement", [0.0, float("nan")], "NaN"),
        ("global_disagreement", [0.0, 1.1], "above"),
    ],
)
def test_invalid_inputs_are_rejected(field: str, value, match: str) -> None:
    kwargs = {
        "sample_keys": ["SET/seed", "SET/item"],
        "features": torch.eye(2),
        "cluster_ids": [0, 0],
        "center_distances": [0.0, 0.1],
        "global_disagreement": [0.0, 0.5],
        "seed_keys": ["SET/seed"],
        "direction": "high",
        "target_counts": (2,),
        "dedup_threshold": 0.95,
    }
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError), match=match):
        build_kmeans_dall_nested_splits(**kwargs)


def test_invalid_direction_and_algorithm_api_are_strict() -> None:
    kwargs = _base_kwargs()
    with pytest.raises(ValueError, match="direction"):
        build_kmeans_dall_nested_splits(**kwargs, direction="middle")

    parameters = inspect.signature(build_kmeans_dall_nested_splits).parameters
    assert "global_disagreement" in parameters
    assert "boundary_disagreement" not in parameters
    assert "scores" not in parameters
