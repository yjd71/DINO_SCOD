from __future__ import annotations

import math

import pytest
import torch

from selection.pc_bpus import greedy_acquire


def test_greedy_updates_novelty_for_repeated_and_orthogonal_prototypes() -> None:
    keys = ["boot", "duplicate", "orthogonal", "mixed"]
    prototypes = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    result = greedy_acquire(
        keys,
        prototypes,
        values=torch.ones(4),
        valid_boundary=torch.ones(4, dtype=torch.bool),
        bootstrap_keys=["boot"],
        target_counts=[1, 3],
    )

    assert result.acquired_keys == ["orthogonal", "mixed"]
    assert result.splits[1] == ["boot"]
    assert result.splits[3] == ["boot", "mixed", "orthogonal"]
    assert result.novelty[0].item() == 1.0
    assert result.novelty[1].item() == pytest.approx(
        1.0 - 1.0 / math.sqrt(2.0), abs=1e-6
    )
    assert result.max_similarity[1].item() == pytest.approx(
        1.0 / math.sqrt(2.0), abs=1e-6
    )
    assert result.utility.dtype == torch.float32
    assert result.utility.device.type == "cpu"


def test_greedy_tie_breaks_by_value_novelty_then_lexical_key() -> None:
    keys = ["boot", "z-candidate", "a-candidate", "high-value"]
    prototypes = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
    )
    result = greedy_acquire(
        keys,
        prototypes,
        values=torch.tensor([0.0, 0.5, 0.5, 0.8]),
        valid_boundary=torch.ones(4, dtype=torch.bool),
        bootstrap_keys=["boot"],
        target_counts=[1, 2],
    )

    # high-value has zero novelty and therefore zero utility.  Equal non-zero
    # utility/value/novelty candidates are finally ordered by sample key.
    assert result.acquired_keys == ["a-candidate"]


def test_greedy_clamps_negative_cosine_and_excludes_invalid_candidates() -> None:
    keys = ["boot", "opposite", "invalid-best"]
    result = greedy_acquire(
        keys,
        prototypes=torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]),
        values=torch.tensor([0.0, 0.4, 1.0]),
        valid_boundary=torch.tensor([False, True, False]),
        bootstrap_keys=["boot"],
        target_counts=[1, 2],
    )

    assert result.acquired_keys == ["opposite"]
    assert result.max_similarity.item() == 0.0
    assert result.novelty.item() == 1.0


def test_greedy_fails_instead_of_filling_from_invalid_candidates() -> None:
    with pytest.raises(RuntimeError, match="Insufficient valid-boundary"):
        greedy_acquire(
            ["boot", "valid", "invalid"],
            prototypes=torch.eye(3),
            values=torch.ones(3),
            valid_boundary=torch.tensor([True, True, False]),
            bootstrap_keys=["boot"],
            target_counts=[1, 3],
        )
