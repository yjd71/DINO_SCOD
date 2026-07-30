from __future__ import annotations

import torch

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.PC_HBM.training.losses import (
    binary_pair_loss,
    pc_hbm_labeled_loss,
    pc_hbm_pc_only_labeled_loss,
    pc_mode_for_epoch,
)
from Model.PC_HBM.training.diagnostics import (
    DIAGNOSTIC_NAMES,
    collect_pc_diagnostics,
)


def _outputs(batch=1):
    return tuple(
        torch.randn(batch, 1, 12, 12, requires_grad=True)
        for _ in range(5)
    )


def _aux(valid=(True, True)):
    logits = torch.tensor(
        [[4.0, -4.0], [-4.0, 4.0]],
        requires_grad=True,
    )
    return {
        "pc_active": True,
        "fallback_reason": None,
        "forward_mode": "full",
        "pc_hbm": {
            "pair_logits": logits,
            "query_valid": torch.tensor(valid),
            "query_batch_ids": torch.tensor([0, 0]),
            "query_flat_indices": torch.tensor([0, 1]),
            "query_mask_map": torch.ones(1, 1, 2, 2),
        },
    }


def test_locked_schedules_and_three_epoch_ramp():
    cfg = DinoPCHBMConfig()
    cfg.configure_training_design("two_stage")
    assert [pc_mode_for_epoch(epoch, cfg) for epoch in (1, 5, 6, 10, 11)] == [
        "off",
        "off",
        "verify_only",
        "verify_only",
        "full",
    ]
    assert [cfg.injection_scale(epoch) for epoch in (11, 12, 13)] == [
        1 / 3,
        2 / 3,
        1.0,
    ]
    cfg.configure_training_design("teacher_only")
    assert [pc_mode_for_epoch(epoch, cfg) for epoch in (1, 5, 6)] == [
        "verify_only",
        "verify_only",
        "full",
    ]


def test_binary_pair_ce_uses_foreground_and_background_targets():
    gt = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    aux = _aux()
    loss, metrics = binary_pair_loss(
        aux, gt, aux["pc_hbm"]["pair_logits"], DinoPCHBMConfig()
    )
    assert float(loss.detach()) < 0.01
    assert float(metrics["pair_accuracy"]) == 1.0
    loss.backward()
    assert torch.isfinite(aux["pc_hbm"]["pair_logits"].grad).all()


def test_fully_invalid_pair_loss_is_finite_differentiable_zero():
    gt = torch.zeros(1, 1, 2, 2)
    aux = _aux(valid=(False, False))
    loss, metrics = binary_pair_loss(
        aux, gt, aux["pc_hbm"]["pair_logits"], DinoPCHBMConfig()
    )
    assert float(loss.detach()) == 0.0
    assert float(metrics["pair_valid_count"]) == 0.0
    loss.backward()
    assert torch.equal(
        aux["pc_hbm"]["pair_logits"].grad,
        torch.zeros_like(aux["pc_hbm"]["pair_logits"]),
    )


def test_two_stage_and_teacher_only_labeled_objectives():
    cfg = DinoPCHBMConfig()
    gt = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    outputs = _outputs()
    aux = _aux()
    two_stage, two_log = pc_hbm_labeled_loss(
        outputs,
        aux,
        gt,
        11,
        cfg,
        pc_mode="full",
        training_design="two_stage",
    )
    assert float(two_log["L_base"]) > 0
    assert float(two_log["L_pair"]) > 0
    assert torch.isfinite(two_stage)

    verify, verify_log = pc_hbm_pc_only_labeled_loss(
        outputs,
        {**aux, "forward_mode": "verify_only"},
        gt,
        1,
        cfg,
        pc_mode="verify_only",
    )
    assert float(verify_log["L_base"]) == 0.0
    assert float(verify_log["L_main"]) == 0.0
    assert torch.allclose(
        verify,
        cfg.lambda_pair * verify_log["L_pair"],
    )

    full, full_log = pc_hbm_pc_only_labeled_loss(
        outputs,
        aux,
        gt,
        6,
        cfg,
        pc_mode="full",
    )
    assert float(full_log["L_main"]) > 0
    assert full > verify


def test_pair_ce_ignores_queries_outside_both_regions():
    cfg = DinoPCHBMConfig()
    logits = torch.randn(1, 2, requires_grad=True)
    aux = {
        "pc_hbm": {
            "pair_logits": logits,
            "query_valid": torch.tensor([True]),
            "query_batch_ids": torch.tensor([0]),
            "query_flat_indices": torch.tensor([5]),
            "query_mask_map": torch.ones(1, 1, 4, 4),
        }
    }
    loss, metrics = binary_pair_loss(
        aux,
        torch.ones(1, 1, 4, 4),
        logits,
        cfg,
    )
    assert float(loss.detach()) == 0.0
    assert float(metrics["pair_valid_count"]) == 0.0


def test_diagnostics_emit_complete_finite_lite_schema():
    aux = {
        "z_main": torch.zeros(1, 1, 4, 4),
        "pc_hbm": {
            "query_valid": torch.tensor([True, True]),
            "query_batch_ids": torch.tensor([0, 0]),
            "query_flat_indices": torch.tensor([0, 1]),
            "query_mask_map": torch.ones(1, 1, 2, 2),
            "pair_logits": torch.tensor([[4.0, -4.0], [-4.0, 4.0]]),
            "region_prob": torch.tensor([[0.8, 0.2], [0.2, 0.8]]),
            "retrieval_valid": torch.ones(2, 2, 2, dtype=torch.bool),
            "parent_cosine": torch.full((2, 2, 2), 0.75),
            "child_cosine": torch.full((2, 2, 2), 0.5),
            "candidate_entropy": torch.full((2, 2), 0.25),
            "memory_confidence": torch.tensor([0.8, 0.7]),
            "gate": torch.tensor([0.4, 0.6]),
            "p3_delta": torch.ones(1, 128, 2, 2) * 0.01,
            "beta": torch.tensor(0.5),
            "route": {"route_entropy_norm": torch.tensor([0.3])},
        },
    }
    gt = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    metrics = collect_pc_diagnostics(
        aux,
        gt,
        pseudo_confidence=torch.full((1, 1, 4, 4), 0.9),
    )
    assert tuple(metrics) == DIAGNOSTIC_NAMES
    assert all(bool(torch.isfinite(value)) for value in metrics.values())
    assert float(metrics["pair_cls_acc"]) == 1.0
