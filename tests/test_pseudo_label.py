from __future__ import annotations

import pytest
import torch

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.PC_HBM.training.pseudo_label import (
    build_pc_confidence,
    confidence_weighted_feature_cosine_loss,
    pc_unlabeled_loss,
    prepare_pseudo_targets,
    weighted_structure_loss,
)


def _teacher_aux():
    probability = torch.tensor([[[[0.9, 0.2], [0.5, 1.0]]]])
    query = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    memory_confidence = torch.tensor([[[[0.25, 0.0], [0.8, 0.0]]]])
    p3 = torch.randn(1, 4, 2, 2)
    return {
        "p_final": probability,
        "pc_active": True,
        "fallback_reason": None,
        "forward_mode": "teacher_pseudo",
        "pc_hbm": {
            "query_mask_map": query,
            "memory_confidence_map": memory_confidence,
            "p3_corr": p3,
        },
        "distill_features": {"p3_corr": p3},
    }


def test_confidence_is_probability_certainty_times_sparse_modifier():
    confidence = build_pc_confidence(_teacher_aux())
    expected = torch.tensor([[[[0.2, 0.6], [0.0, 1.0]]]])
    assert torch.allclose(confidence, expected)


def test_memory_confidence_upsamples_bilinearly():
    probability = torch.full((1, 1, 4, 4), 0.9)
    memory_map = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]])
    aux = {
        "p_final": probability,
        "pc_active": True,
        "forward_mode": "teacher_pseudo",
        "fallback_reason": None,
        "pc_hbm": {
            "query_mask_map": torch.ones(1, 1, 2, 2),
            "memory_confidence_map": memory_map,
        },
    }
    expected = 0.8 * torch.nn.functional.interpolate(
        memory_map,
        size=(4, 4),
        mode="bilinear",
        align_corners=False,
    )
    assert torch.allclose(build_pc_confidence(aux), expected)


def test_pseudo_targets_contain_only_detached_soft_mask_and_confidence():
    targets = prepare_pseudo_targets(_teacher_aux())
    assert set(targets) == {"p_soft", "confidence"}
    for value in targets.values():
        assert not value.requires_grad


def test_pseudo_targets_created_in_inference_mode_are_backward_safe():
    with torch.inference_mode():
        targets = prepare_pseudo_targets(_teacher_aux())

    assert all(not torch.is_inference(value) for value in targets.values())
    outputs = tuple(
        torch.randn(1, 1, 2, 2, requires_grad=True) for _ in range(5)
    )
    aux = {
        "forward_mode": "full",
        "pc_active": True,
        "pc_engine_source": "external_readonly",
        "z_main": outputs[3],
    }
    loss, _ = pc_unlabeled_loss(
        outputs,
        aux,
        targets["p_soft"],
        targets["confidence"],
        DinoPCHBMConfig(),
    )
    loss.backward()
    assert all(output.grad is not None for output in outputs)


def test_pseudo_targets_do_not_require_distillation_features():
    aux = _teacher_aux()
    aux["distill_features"] = None
    assert set(prepare_pseudo_targets(aux)) == {"p_soft", "confidence"}


def test_p3_cosine_distillation_is_student_only():
    student = torch.randn(2, 8, 4, 4, requires_grad=True)
    teacher = torch.randn(2, 8, 4, 4, requires_grad=True)
    confidence = torch.ones(2, 1, 8, 8)
    loss = confidence_weighted_feature_cosine_loss(
        student, teacher, confidence
    )
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_unlabeled_loss_has_only_confidence_weighted_mask_terms():
    cfg = DinoPCHBMConfig()
    outputs = tuple(
        torch.randn(2, 1, 8, 8, requires_grad=True) for _ in range(5)
    )
    aux = {
        "forward_mode": "full",
        "pc_active": True,
        "pc_engine_source": "external_readonly",
        "z_main": outputs[3],
    }
    pseudo = torch.rand(2, 1, 8, 8)
    confidence = torch.rand(2, 1, 8, 8)
    loss, metrics = pc_unlabeled_loss(
        outputs,
        aux,
        pseudo,
        confidence,
        cfg,
    )
    assert set(metrics) == {
        "L_u_main",
        "L_u_side",
        "L_u_pseudo",
        "L_u_total",
        "pseudo_conf_mean",
        "pseudo_coverage",
    }
    torch.testing.assert_close(
        metrics["L_u_total"],
        cfg.lambda_u * metrics["L_u_pseudo"],
    )
    loss.backward()
    assert all(output.grad is not None for output in outputs)


def test_zero_confidence_returns_finite_differentiable_zero():
    logits = torch.randn(1, 1, 8, 8, requires_grad=True)
    loss = weighted_structure_loss(
        logits,
        torch.rand_like(logits),
        torch.zeros_like(logits),
    )
    assert torch.isfinite(loss)
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert torch.equal(logits.grad, torch.zeros_like(logits))
