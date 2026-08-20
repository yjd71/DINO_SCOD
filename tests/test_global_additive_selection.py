from __future__ import annotations

import pytest
import torch

from utils.global_additive import (
    GLOBAL_ADDITIVE_DEDUP_PROTOCOL_VERSION,
    GLOBAL_ADDITIVE_FORMULA_VERSION,
    build_global_deduplicated_nested_splits,
    build_global_nested_splits,
    compute_global_additive_score,
    labeled_names_from_keys,
    normalize_global_dino_features,
)


def test_additive_score_is_exact_float32_cpu_and_preserves_values_above_one() -> None:
    result = compute_global_additive_score(
        torch.tensor([0.0, 0.25, 1.0], dtype=torch.float64),
        torch.tensor([0.0, 0.5, 0.0], dtype=torch.float64),
    )

    assert GLOBAL_ADDITIVE_FORMULA_VERSION == "global_additive_v1_replicate_hypot"
    assert result.dtype == torch.float32
    assert result.device.type == "cpu"
    assert torch.equal(result, torch.tensor([1.0, 0.75, 2.0]))
    assert result[-1].item() == 2.0


def test_identical_views_have_score_one() -> None:
    result = compute_global_additive_score(torch.zeros(3), torch.zeros(3))

    assert torch.equal(result, torch.ones(3))


def test_constant_global_disagreement_uses_one_minus_term_exactly() -> None:
    result = compute_global_additive_score(
        torch.tensor([0.0, 0.0]),
        torch.tensor([0.4, 0.6]),
    )

    torch.testing.assert_close(result, torch.tensor([0.6, 0.4]))


@pytest.mark.parametrize(
    ("boundary", "global_value", "message"),
    [
        (torch.zeros(2, 1), torch.zeros(2, 1), "one-dimensional"),
        (torch.zeros(2), torch.zeros(3), "same shape"),
        (torch.tensor([float("nan")]), torch.zeros(1), "NaN or infinity"),
        (torch.zeros(1), torch.tensor([float("inf")]), "NaN or infinity"),
        (torch.tensor([-0.01]), torch.zeros(1), r"\[0, 1\]"),
        (torch.zeros(1), torch.tensor([1.01]), r"\[0, 1\]"),
    ],
)
def test_additive_score_rejects_invalid_shape_finite_and_range(
    boundary: torch.Tensor,
    global_value: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_global_additive_score(boundary, global_value)


def test_global_score_ties_use_lexical_key_and_exclude_seed_from_global_rank() -> None:
    keys = ["TR-CAMO/z", "TR-CAMO/c", "TR-CAMO/a", "TR-CAMO/b"]
    result = build_global_nested_splits(
        keys,
        torch.ones(4),
        ["TR-CAMO/z"],
        target_counts=(2, 3),
    )

    assert result.splits[2] == ["TR-CAMO/a", "TR-CAMO/z"]
    assert result.splits[3] == ["TR-CAMO/a", "TR-CAMO/b", "TR-CAMO/z"]
    assert result.selection_order == ("TR-CAMO/z", "TR-CAMO/a", "TR-CAMO/b")
    assert result.global_rank == {
        "TR-CAMO/a": 1,
        "TR-CAMO/b": 2,
        "TR-CAMO/c": 3,
    }
    assert result.selection_rank[2] == {"TR-CAMO/a": 2, "TR-CAMO/z": 1}


def test_41_202_404_808_splits_are_exact_nested_and_inherit_all_seeds() -> None:
    sample_count = 900
    keys = [f"TR-CAMO/sample_{index:04d}" for index in range(sample_count)]
    seeds = keys[::22][:40]
    scores = torch.linspace(0.0, 2.0, sample_count)

    first = build_global_nested_splits(keys, scores, seeds)
    second = build_global_nested_splits(keys, scores, seeds)

    assert {target: len(split) for target, split in first.splits.items()} == {
        41: 41,
        202: 202,
        404: 404,
        808: 808,
    }
    assert set(first.splits[41]) < set(first.splits[202])
    assert set(first.splits[202]) < set(first.splits[404])
    assert set(first.splits[404]) < set(first.splits[808])
    assert set(seeds).issubset(first.splits[41])
    assert len(first.selection_order) == 808
    assert len(first.global_rank) == sample_count - len(seeds)
    assert set(first.selection_rank) == {41, 202, 404, 808}
    assert set(first.selection_rank[808]) == set(first.splits[808])
    assert first == second


def test_labeled_names_preserve_key_order_strip_suffix_and_reject_duplicate_stems() -> None:
    assert labeled_names_from_keys(
        ["TR-CAMO/c", "TR-COD10K/a", "TR-CAMO/animal.v2"]
    ) == ["c", "a", "animal.v2"]

    with pytest.raises(ValueError, match="duplicate labeled stems"):
        labeled_names_from_keys(["TR-CAMO/same", "TR-COD10K/same"])


def test_global_selection_rejects_missing_seed_and_invalid_targets() -> None:
    keys = ["TR-CAMO/a", "TR-CAMO/b", "TR-CAMO/c"]
    scores = torch.tensor([1.0, 0.5, 0.0])

    with pytest.raises(ValueError, match="absent from the catalog"):
        build_global_nested_splits(keys, scores, ["TR-CAMO/missing"], (2,))
    with pytest.raises(ValueError, match="strictly increasing"):
        build_global_nested_splits(keys, scores, ["TR-CAMO/a"], (3, 2))
    with pytest.raises(ValueError, match=r"\[0, 2\]"):
        build_global_nested_splits(keys, [2.1, 0.5, 0.0], ["TR-CAMO/a"], (2,))


def test_global_dino_features_are_float32_cpu_and_l2_normalized() -> None:
    normalized = normalize_global_dino_features(
        torch.tensor([[3.0, 4.0], [5.0, 12.0]], dtype=torch.float64),
        expected_length=2,
    )

    assert normalized.dtype == torch.float32
    assert normalized.device.type == "cpu"
    assert torch.allclose(torch.linalg.vector_norm(normalized, dim=1), torch.ones(2))


def test_dedup_threshold_equality_is_strictly_accepted() -> None:
    keys = ["TR-CAMO/seed", "TR-CAMO/candidate"]
    result = build_global_deduplicated_nested_splits(
        keys,
        scores=torch.tensor([0.0, 2.0]),
        seed_keys=["TR-CAMO/seed"],
        features=torch.tensor([[1.0, 0.0], [0.5, 0.8660254]]),
        target_counts=(2,),
        dedup_threshold=0.5,
    )

    decision = result.audit["TR-CAMO/candidate"]
    assert result.splits[2] == ["TR-CAMO/candidate", "TR-CAMO/seed"]
    assert decision.decision == "strict_selected"
    assert decision.max_cosine_similarity == pytest.approx(0.5)
    assert decision.reference_key == "TR-CAMO/seed"
    assert decision.relaxed is False


def test_dedup_compares_with_seed_and_samples_selected_by_smaller_split() -> None:
    keys = [
        "TR-CAMO/seed",
        "TR-CAMO/a",
        "TR-CAMO/b",
        "TR-CAMO/c",
        "TR-CAMO/d",
    ]
    result = build_global_deduplicated_nested_splits(
        keys,
        scores=torch.tensor([0.0, 2.0, 1.9, 1.8, 1.7]),
        seed_keys=["TR-CAMO/seed"],
        features=torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        target_counts=(2, 3),
        dedup_threshold=0.95,
    )

    assert result.splits[2] == ["TR-CAMO/a", "TR-CAMO/seed"]
    assert result.splits[3] == ["TR-CAMO/a", "TR-CAMO/c", "TR-CAMO/seed"]
    assert set(result.splits[2]) < set(result.splits[3])
    assert result.audit["TR-CAMO/seed"].decision == "seed"
    duplicate = result.audit["TR-CAMO/b"]
    assert duplicate.decision == "duplicate_skipped"
    assert duplicate.max_cosine_similarity == pytest.approx(1.0)
    assert duplicate.reference_key == "TR-CAMO/a"
    assert duplicate.evaluated_target_count == 3
    assert result.audit["TR-CAMO/d"].decision == "not_evaluated"
    assert result.rounds[0].strict_selected_count == 1
    assert result.rounds[1].strict_selected_count == 1
    assert result.rounds[1].skipped_count == 1
    assert result.rounds[1].cumulative_skipped_count == 1


def test_dedup_relaxed_backfill_uses_original_global_rank_and_is_audited() -> None:
    keys = [
        "TR-CAMO/seed",
        "TR-CAMO/a",
        "TR-CAMO/b",
        "TR-CAMO/c",
        "TR-CAMO/d",
    ]
    result = build_global_deduplicated_nested_splits(
        keys,
        scores=torch.tensor([0.0, 2.0, 1.5, 1.0, 0.5]),
        seed_keys=["TR-CAMO/seed"],
        features=torch.ones(5, 3),
        target_counts=(2, 4),
        dedup_threshold=0.95,
    )

    assert result.protocol_version == GLOBAL_ADDITIVE_DEDUP_PROTOCOL_VERSION
    assert result.dedup_threshold == pytest.approx(0.95)
    assert result.selection_order == (
        "TR-CAMO/seed",
        "TR-CAMO/a",
        "TR-CAMO/b",
        "TR-CAMO/c",
    )
    assert result.splits[2] == ["TR-CAMO/a", "TR-CAMO/seed"]
    assert result.splits[4] == [
        "TR-CAMO/a",
        "TR-CAMO/b",
        "TR-CAMO/c",
        "TR-CAMO/seed",
    ]
    assert set(result.splits[2]) < set(result.splits[4])
    assert [result.audit[key].decision for key in keys[1:4]] == [
        "relaxed_backfill",
        "relaxed_backfill",
        "relaxed_backfill",
    ]
    assert all(result.audit[key].relaxed for key in keys[1:4])
    assert result.audit["TR-CAMO/d"].decision == "duplicate_skipped"
    assert result.audit["TR-CAMO/d"].max_cosine_similarity == pytest.approx(1.0)
    assert result.audit["TR-CAMO/d"].reference_key == "TR-CAMO/seed"

    first_round, second_round = result.rounds
    assert (first_round.strict_selected_count, first_round.skipped_count) == (0, 4)
    assert first_round.relaxed_selected_count == 1
    assert first_round.cumulative_skipped_count == 4
    assert first_round.cumulative_relaxed_selected_count == 1
    assert (second_round.strict_selected_count, second_round.skipped_count) == (0, 0)
    assert second_round.relaxed_selected_count == 2
    assert second_round.cumulative_strict_selected_count == 0
    assert second_round.cumulative_skipped_count == 4
    assert second_round.cumulative_relaxed_selected_count == 3


def test_deduplicated_41_202_404_808_splits_are_exact_and_nested() -> None:
    sample_count = 900
    keys = [f"TR-CAMO/sample_{index:04d}" for index in range(sample_count)]
    seeds = keys[:40]
    result = build_global_deduplicated_nested_splits(
        keys,
        scores=torch.linspace(0.0, 2.0, sample_count),
        seed_keys=seeds,
        features=torch.ones(sample_count, 1),
        dedup_threshold=1.0,
    )

    assert {target: len(split) for target, split in result.splits.items()} == {
        41: 41,
        202: 202,
        404: 404,
        808: 808,
    }
    assert set(seeds).issubset(result.splits[41])
    assert set(result.splits[41]) < set(result.splits[202])
    assert set(result.splits[202]) < set(result.splits[404])
    assert set(result.splits[404]) < set(result.splits[808])
    assert all(round_stats.skipped_count == 0 for round_stats in result.rounds)
    assert all(round_stats.relaxed_selected_count == 0 for round_stats in result.rounds)


@pytest.mark.parametrize(
    ("features", "message"),
    [
        (torch.ones(3), r"\[N, D\]"),
        (torch.ones(2, 3), "3 rows"),
        (torch.tensor([[1.0, 0.0], [float("nan"), 1.0], [1.0, 0.0]]), "NaN"),
        (torch.tensor([[1.0, 0.0], [float("inf"), 1.0], [1.0, 0.0]]), "infinity"),
        (torch.tensor([[1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]), "zero-norm"),
        (torch.empty(3, 0), "non-empty feature dimension"),
    ],
)
def test_dedup_rejects_bad_features(features: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_global_deduplicated_nested_splits(
            ["TR-CAMO/seed", "TR-CAMO/a", "TR-CAMO/b"],
            scores=torch.tensor([0.0, 1.0, 0.5]),
            seed_keys=["TR-CAMO/seed"],
            features=features,
            target_counts=(2,),
        )
