from __future__ import annotations

import pytest
import torch

from selection.bpus_v2 import greedy_acquire_bpus_v2


def _prototypes(rows: list[dict[int, float]]) -> torch.Tensor:
    tensor = torch.zeros(len(rows), 128)
    for row, values in enumerate(rows):
        for column, value in values.items():
            tensor[row, column] = value
    return tensor


def test_soft_reward_keeps_value_when_novelty_is_zero_and_updates_similarity() -> None:
    keys = ["bootstrap", "repeat", "orthogonal", "repeat_second"]
    prototypes = _prototypes([{0: 1.0}, {0: 1.0}, {1: 1.0}, {1: 1.0}])
    result = greedy_acquire_bpus_v2(
        keys,
        prototypes,
        values=[0.1, 0.9, 0.8, 0.7],
        valid_boundary=[True, True, True, True],
        bootstrap_keys=["bootstrap"],
        target_counts=[1, 4],
    )

    assert result.acquired_keys == ["orthogonal", "repeat", "repeat_second"]
    assert result.novelty.tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert result.utility.tolist() == pytest.approx([1.6, 0.9, 0.7])
    assert result.splits[4] == sorted(keys)


def test_soft_reward_can_prefer_higher_value_over_more_novel_candidate() -> None:
    result = greedy_acquire_bpus_v2(
        ["boot", "same_high", "novel_low"],
        _prototypes([{0: 1.0}, {0: 1.0}, {1: 1.0}]),
        values=[0.0, 0.9, 0.4],
        valid_boundary=[True, True, True],
        bootstrap_keys=["boot"],
        target_counts=[1, 2],
    )
    assert result.acquired_keys == ["same_high"]
    assert result.utility.item() == pytest.approx(0.9)


def test_tie_breaks_by_value_novelty_then_lexical_key() -> None:
    result = greedy_acquire_bpus_v2(
        ["boot", "z", "a"],
        _prototypes([{0: 1.0}, {1: 1.0}, {1: 1.0}]),
        values=[0.0, 0.5, 0.5],
        valid_boundary=[True, True, True],
        bootstrap_keys=["boot"],
        target_counts=[1, 2],
    )
    assert result.acquired_keys == ["a"]


def test_negative_cosine_is_clamped_and_invalid_candidates_are_excluded() -> None:
    result = greedy_acquire_bpus_v2(
        ["boot", "negative", "invalid"],
        _prototypes([{0: 1.0}, {0: -1.0}, {1: 1.0}]),
        values=[0.0, 0.2, 1.0],
        valid_boundary=[True, True, False],
        bootstrap_keys=["boot"],
        target_counts=[1, 2],
    )
    assert result.acquired_keys == ["negative"]
    assert result.novelty.item() == 1.0
    assert result.max_similarity.item() == 0.0


def test_insufficient_valid_candidates_fails_without_fallback() -> None:
    with pytest.raises(RuntimeError, match="Insufficient valid-boundary candidates"):
        greedy_acquire_bpus_v2(
            ["boot", "valid", "invalid"],
            _prototypes([{0: 1.0}, {1: 1.0}, {2: 1.0}]),
            values=[0.0, 0.5, 1.0],
            valid_boundary=[True, True, False],
            bootstrap_keys=["boot"],
            target_counts=[1, 3],
        )
