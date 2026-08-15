from __future__ import annotations

import torch

from Model.PC_HBM.training.diagnostics import (
    _tie_aware_binary_auroc,
    collect_pc_diagnostics,
)
from tools.evaluate_child_verification import (
    exact_tie_aware_auroc,
    same_region_derangement,
)


def test_verification_diagnostics_report_repairs_auroc_and_margins() -> None:
    verify_logits = torch.tensor(
        [
            [[3.0, 3.0], [-3.0, -3.0]],
            [[-2.0, -2.0], [2.0, 2.0]],
        ]
    )
    parent_scores = torch.tensor(
        [
            [[-1.0, -1.0], [1.0, 1.0]],
            [[-1.0, -1.0], [1.0, 1.0]],
        ]
    )
    candidate_valid = torch.ones(2, 2, 2, dtype=torch.bool)
    relation_valid = candidate_valid.clone()
    relation_valid[0, 0, 0] = False
    aux = {
        "z_main": torch.zeros(1, 1, 4, 4),
        "pc_hbm": {
            "query_valid": torch.tensor([True, True]),
            "query_batch_ids": torch.tensor([0, 0]),
            "query_flat_indices": torch.tensor([0, 1]),
            "query_mask_map": torch.ones(1, 1, 2, 2),
            "pair_logits": torch.tensor([[2.0, -2.0], [-2.0, 2.0]]),
            "parent_region_logits": torch.tensor([[-1.0, 1.0], [-1.0, 1.0]]),
            "region_prob": torch.tensor([[0.9, 0.1], [0.1, 0.9]]),
            "retrieval_valid": candidate_valid,
            "parent_cosine": parent_scores,
            "parent_scores": parent_scores,
            "child_cosine": torch.ones_like(parent_scores),
            "child_match_logits": verify_logits,
            "child_verify_logits": verify_logits,
            "relation_valid": relation_valid,
            "child_match_strength": torch.tensor(0.25),
            "verification_strength": torch.tensor(0.25),
            "candidate_entropy": torch.zeros(2),
            "memory_confidence": torch.ones(2, 1),
            "gate": torch.ones(2, 1),
            "p3_delta": torch.zeros(2, 128),
            "beta": torch.tensor(0.0),
        },
    }
    gt = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])

    metrics = collect_pc_diagnostics(aux, gt)

    assert metrics["pair_cls_acc"] == 1.0
    assert metrics["parent_only_accuracy"] == 0.5
    assert metrics["verified_accuracy"] == 1.0
    assert metrics["verification_repair_rate"] == 0.5
    assert metrics["verification_harm_rate"] == 0.0
    assert metrics["verification_net_gain"] == 0.5
    assert metrics["margin_gain_parent_wrong"] > 0.0
    assert metrics["candidate_auroc"] == 1.0
    assert metrics["relation_valid_ratio"] == 7.0 / 8.0
    assert metrics["child_match_strength"] == 0.25
    assert metrics["verification_strength"] == 0.25
    assert metrics["child_match_logit_mean"] == metrics["verify_logit_mean"]
    assert metrics["child_match_logit_std"] == metrics["verify_logit_std"]
    assert metrics["child_update_parent_ratio"] == metrics[
        "verification_update_parent_ratio"
    ]
    assert torch.isfinite(metrics["verify_logit_mean"])
    assert torch.isfinite(metrics["verify_logit_std"])
    assert torch.isfinite(metrics["verification_update_parent_ratio"])
    assert "verification_abs_weight" not in metrics
    assert "verification_rel_weight" not in metrics
    assert "verification_bias" not in metrics


def test_exact_auroc_is_tie_aware() -> None:
    assert exact_tie_aware_auroc(
        torch.tensor([0.0, 0.0]), torch.tensor([True, False])
    ) == 0.5
    assert exact_tie_aware_auroc(
        torch.tensor([2.0, 1.0, -1.0]),
        torch.tensor([True, True, False]),
    ) == 1.0
    assert exact_tie_aware_auroc(
        torch.tensor([1.0]), torch.tensor([True])
    ) == 0.0


def test_exact_auroc_matches_pairwise_reference_with_ties() -> None:
    torch.manual_seed(23)
    scores = torch.randint(-3, 4, (257,), dtype=torch.float32)
    targets = torch.rand(257) > 0.55
    positives = scores[targets]
    negatives = scores[~targets]
    differences = positives[:, None] - negatives[None, :]
    reference = (
        (differences > 0.0).float()
        + 0.5 * (differences == 0.0).float()
    ).mean()

    assert exact_tie_aware_auroc(scores, targets) == reference.item()
    torch.testing.assert_close(
        _tie_aware_binary_auroc(scores, targets, torch.tensor(0.0)),
        reference,
    )


def test_exact_auroc_handles_large_candidate_batches_without_quadratic_memory() -> None:
    scores = torch.arange(100_000, dtype=torch.float32).remainder(101)
    targets = torch.arange(100_000).remainder(3) == 0

    result = exact_tie_aware_auroc(scores, targets)
    training_result = _tie_aware_binary_auroc(
        scores, targets, torch.tensor(0.0)
    )

    assert 0.0 <= result <= 1.0
    assert training_result.item() == result


def test_same_region_shuffle_is_seeded_deranged_and_mask_preserving() -> None:
    child_keys = torch.arange(8.0).reshape(1, 2, 4, 1)
    valid = torch.tensor([[[True, True, True, False], [True, False, False, False]]])

    shuffled, counts = same_region_derangement(child_keys, valid, seed=11)
    repeated, repeated_counts = same_region_derangement(child_keys, valid, seed=11)

    assert torch.equal(shuffled, repeated)
    assert counts == repeated_counts
    assert counts == {
        "eligible_candidates": 3,
        "valid_candidates": 4,
        "shuffled_groups": 1,
    }
    assert torch.equal(
        shuffled[0, 0, valid[0, 0]].flatten().sort().values,
        child_keys[0, 0, valid[0, 0]].flatten().sort().values,
    )
    assert not torch.any(
        shuffled[0, 0, valid[0, 0]] == child_keys[0, 0, valid[0, 0]]
    )
    assert torch.equal(shuffled[~valid], child_keys[~valid])
    assert torch.equal(shuffled[0, 1], child_keys[0, 1])
