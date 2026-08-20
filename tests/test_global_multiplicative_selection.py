from __future__ import annotations

import pytest
import torch

from utils.global_multiplicative import (
    GLOBAL_MULTIPLICATIVE_FORMULA_VERSION,
    build_global_multiplicative_splits,
    compute_global_multiplicative_score,
    labeled_names_from_keys,
)


def test_multiplicative_score_is_exact_float32_cpu() -> None:
    boundary = torch.tensor([0.0, 0.4, 1.0, 0.75], dtype=torch.float32)
    global_value = torch.tensor([0.0, 0.25, 1.0, 0.2], dtype=torch.float32)

    result = compute_global_multiplicative_score(boundary, global_value)

    assert GLOBAL_MULTIPLICATIVE_FORMULA_VERSION.startswith("global_multiplicative_v1")
    assert result.dtype == torch.float32
    assert result.device.type == "cpu"
    assert torch.equal(result, boundary * (1.0 - global_value))
    assert result.tolist() == pytest.approx([0.0, 0.3, 0.0, 0.6])


def test_identical_views_have_zero_multiplicative_score() -> None:
    zeros = torch.zeros(3, dtype=torch.float32)
    assert torch.equal(compute_global_multiplicative_score(zeros, zeros), zeros)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_multiplicative_score_rejects_non_float32_components(
    dtype: torch.dtype,
) -> None:
    with pytest.raises(TypeError, match="torch.float32"):
        compute_global_multiplicative_score(
            torch.tensor([0.2], dtype=dtype),
            torch.tensor([0.1], dtype=torch.float32),
        )


@pytest.mark.parametrize(
    ("boundary", "global_value", "message"),
    [
        (torch.zeros((1, 1)), torch.zeros(1), "one-dimensional"),
        (torch.zeros(2), torch.zeros(1), "same shape"),
        (torch.tensor([float("nan")]), torch.zeros(1), "NaN or infinity"),
        (torch.tensor([1.1]), torch.zeros(1), r"\[0, 1\]"),
        (torch.zeros(1), torch.tensor([float("inf")]), "NaN or infinity"),
    ],
)
def test_multiplicative_score_rejects_invalid_inputs(
    boundary: torch.Tensor,
    global_value: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        compute_global_multiplicative_score(
            boundary.to(dtype=torch.float32),
            global_value.to(dtype=torch.float32),
        )


def test_global_ties_use_lexical_key_and_seed_is_excluded_from_rank() -> None:
    keys = ["TR-CAMO/c", "TR-CAMO/a", "TR-CAMO/seed", "TR-CAMO/b"]
    scores = torch.tensor([0.5, 0.5, 1.0, 0.5], dtype=torch.float32)

    result = build_global_multiplicative_splits(
        keys,
        scores,
        ["TR-CAMO/seed"],
        target_counts=(2, 4),
    )

    assert result.global_rank == {
        "TR-CAMO/a": 1,
        "TR-CAMO/b": 2,
        "TR-CAMO/c": 3,
    }
    assert "TR-CAMO/seed" not in result.global_rank
    assert result.selection_order == (
        "TR-CAMO/seed",
        "TR-CAMO/a",
        "TR-CAMO/b",
        "TR-CAMO/c",
    )


def test_formal_splits_are_exact_nested_and_inherit_all_seeds() -> None:
    keys = [f"TR-COD10K/image_{index:04d}" for index in range(900)]
    scores = torch.linspace(0.0, 0.999, len(keys), dtype=torch.float32)
    seeds = [keys[index] for index in range(0, 800, 20)]
    targets = (41, 202, 404, 808)

    result = build_global_multiplicative_splits(
        keys,
        scores,
        seeds,
        target_counts=targets,
    )

    previous: set[str] | None = None
    for target in targets:
        current = set(result.splits[target])
        assert len(result.splits[target]) == target
        assert set(seeds).issubset(current)
        if previous is not None:
            assert previous.issubset(current)
        previous = current


def test_labeled_names_follow_pt_order_and_reject_duplicate_stems() -> None:
    keys = ["TR-CAMO/a", "TR-COD10K/b.v2"]
    assert labeled_names_from_keys(keys) == ["a", "b.v2"]
    with pytest.raises(ValueError, match="duplicate labeled stems"):
        labeled_names_from_keys(["TR-CAMO/a", "TR-COD10K/a"])


def test_selection_rejects_missing_seed_bad_scores_and_invalid_targets() -> None:
    keys = ["TR-CAMO/a", "TR-CAMO/b"]
    scores = torch.tensor([0.2, 0.1], dtype=torch.float32)
    with pytest.raises(ValueError, match="absent"):
        build_global_multiplicative_splits(keys, scores, ["TR-CAMO/missing"], (1,))
    with pytest.raises(ValueError, match="2 elements"):
        build_global_multiplicative_splits(
            keys, torch.tensor([0.2], dtype=torch.float32), ["TR-CAMO/a"], (2,)
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        build_global_multiplicative_splits(keys, scores, ["TR-CAMO/a"], (2, 2))
