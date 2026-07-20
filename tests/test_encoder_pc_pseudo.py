from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from Model.PC_HBM.training.encoder_pseudo import (
    build_encoder_pc_confidence,
    confidence_weighted_logit_bce,
    encoder_pc_unlabeled_loss,
    prepare_encoder_pc_pseudo_targets,
)


def _config(**overrides):
    values = {
        "route_confidence_floor": 0.20,
        "pseudo_fg_threshold": 0.70,
        "pseudo_bg_threshold": 0.30,
        "pseudo_hard_ramp_epochs": 3,
        "hard_coverage_target": 0.20,
        "hard_loss_weight": 2.0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _teacher_payload(*, route_entropy: float = 0.0, route_confidence: float = 0.4):
    p_refined = torch.tensor([[[[0.8, 0.2], [0.9, 0.1]]]])
    p_core = torch.tensor([[[[0.7, 0.3], [0.6, 0.4]]]])
    pi = torch.tensor(
        [
            [
                [[0.7, 0.7], [0.7, 0.7]],
                [[0.1, 0.1], [0.1, 0.1]],
                [[0.1, 0.1], [0.1, 0.1]],
                [[0.1, 0.1], [0.1, 0.1]],
            ]
        ]
    )
    return {
        "z_core": torch.logit(p_core),
        "pseudo_refiner": {
            "p_pseudo_refined": p_refined,
            "pi": pi,
        },
        "encoder_pc_hbm": {
            "C23_map": torch.full_like(p_refined, 0.1),
            "route": {
                "route_confidence": torch.tensor([route_confidence]),
                # This legacy diagnostic must have no effect on confidence.
                "route_entropy_norm": torch.tensor([route_entropy]),
            },
        },
    }


def _student_contract(batch: int = 1, height: int = 4, width: int = 4):
    outputs = tuple(
        torch.zeros(batch, 1, height, width, requires_grad=True) for _ in range(5)
    )
    aux_core = torch.ones_like(outputs[3], requires_grad=True)
    aux = {"z_core": aux_core, "pseudo_refiner": None}
    return outputs, aux, aux_core


def test_confidence_is_exact_five_factor_product_and_ignores_route_entropy():
    payload = _teacher_payload(route_entropy=0.0)
    confidence = build_encoder_pc_confidence(payload, _config())
    payload_high_entropy = _teacher_payload(route_entropy=1.0)
    high_entropy_confidence = build_encoder_pc_confidence(
        payload_high_entropy, _config()
    )

    p_refined = payload["pseudo_refiner"]["p_pseudo_refined"]
    p_core = torch.sigmoid(payload["z_core"])
    pi = payload["pseudo_refiner"]["pi"]
    entropy = -(pi * pi.clamp_min(1.0e-8).log()).sum(1, keepdim=True) / math.log(4.0)
    expected = (
        (2.0 * (p_refined - 0.5).abs())
        * 0.9
        * (1.0 - entropy)
        * 0.4
        * (0.5 + 0.5 * torch.exp(-(p_refined - p_core).abs() / 0.25))
    )
    assert torch.allclose(confidence, expected, atol=1.0e-6, rtol=1.0e-6)
    assert torch.equal(confidence, high_entropy_confidence)
    assert not confidence.requires_grad


def test_route_confidence_floor_is_kept_without_entropy_multiplier():
    low_route = build_encoder_pc_confidence(
        _teacher_payload(route_entropy=1.0, route_confidence=0.01),
        _config(),
    )
    floor_route = build_encoder_pc_confidence(
        _teacher_payload(route_entropy=0.0, route_confidence=0.20),
        _config(),
    )
    assert torch.equal(low_route, floor_route)
    assert torch.count_nonzero(low_route) == low_route.numel()


def test_prepare_targets_detaches_refiner_and_applies_hard_thresholds():
    payload = _teacher_payload()
    payload["pseudo_refiner"]["p_pseudo_refined"].requires_grad_(True)
    pseudo = prepare_encoder_pc_pseudo_targets(payload, _config())

    assert torch.equal(
        pseudo["hard_target"],
        torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]]),
    )
    assert pseudo["hard_valid"].all()
    assert torch.equal(
        pseudo["hard_weight"],
        pseudo["confidence"] * pseudo["hard_valid"],
    )
    assert pseudo["hard_coverage"].item() == pytest.approx(1.0)
    assert pseudo["hard_coverage_scale"].item() == pytest.approx(1.0)
    assert not pseudo["p_soft"].requires_grad
    assert not pseudo["confidence"].requires_grad


def test_weighted_bce_consumes_logits_and_empty_weight_is_differentiable_zero():
    logits = torch.zeros(1, 1, 2, 2, requires_grad=True)
    target = torch.ones_like(logits)
    one = torch.ones_like(logits)
    loss = confidence_weighted_logit_bce(logits, target, one)
    assert loss.item() == pytest.approx(math.log(2.0), rel=1.0e-6)
    loss.backward()
    assert torch.all(logits.grad < 0)

    empty_logits = torch.randn(1, 1, 2, 2, requires_grad=True)
    empty = confidence_weighted_logit_bce(
        empty_logits,
        target,
        torch.zeros_like(target),
    )
    assert empty.item() == pytest.approx(0.0)
    empty.backward()
    assert empty_logits.grad is not None
    assert torch.count_nonzero(empty_logits.grad) == 0


def test_unlabeled_loss_supervises_outputs_three_not_aux_core():
    outputs, aux, aux_core = _student_contract()
    pseudo = {
        "p_soft": torch.full((1, 1, 4, 4), 0.8),
        "confidence": torch.ones(1, 1, 4, 4),
    }
    total, log = encoder_pc_unlabeled_loss(outputs, aux, pseudo, _config(), 1)
    total.backward()

    assert outputs[3].grad is not None
    assert torch.count_nonzero(outputs[3].grad) > 0
    assert aux_core.grad is None
    assert log["hard_ramp"].item() == pytest.approx(1.0 / 3.0)


def test_hard_coverage_scale_and_three_epoch_ramp_are_applied_exactly():
    outputs, aux, _ = _student_contract(height=1, width=10)
    # One of ten pixels is hard-valid, so coverage=.10 and scale=.10/.20=.50.
    p_soft = torch.full((1, 1, 1, 10), 0.5)
    p_soft[..., 0] = 0.8
    pseudo = {"p_soft": p_soft, "confidence": torch.ones_like(p_soft)}
    total, log = encoder_pc_unlabeled_loss(outputs, aux, pseudo, _config(), 2)

    assert log["hard_valid_ratio"].item() == pytest.approx(0.10)
    assert log["hard_coverage_scale"].item() == pytest.approx(0.50)
    assert log["hard_ramp"].item() == pytest.approx(2.0 / 3.0)
    expected_hard = 2.0 * (2.0 / 3.0) * 0.5 * log["L_u_hard"]
    assert torch.allclose(log["L_u_hard_scaled"], expected_hard)
    assert torch.allclose(
        total.detach(),
        log["L_u_soft"] + log["L_u_hard_scaled"] + log["L_u_side"],
    )


def test_no_reliable_confidence_returns_differentiable_zero_for_all_outputs():
    outputs, aux, _ = _student_contract()
    pseudo = {
        "p_soft": torch.full((1, 1, 4, 4), 0.5),
        "confidence": torch.zeros(1, 1, 4, 4),
    }
    total, log = encoder_pc_unlabeled_loss(outputs, aux, pseudo, _config(), 3)
    assert total.item() == pytest.approx(0.0)
    assert log["hard_valid_ratio"].item() == pytest.approx(0.0)
    assert log["hard_coverage_scale"].item() == pytest.approx(0.0)
    total.backward()
    for output in outputs:
        assert output.grad is not None
        assert torch.count_nonzero(output.grad) == 0


def test_student_unlabeled_rejects_any_refiner_execution():
    outputs, aux, _ = _student_contract()
    aux["pseudo_refiner"] = {"p_pseudo_refined": torch.ones(1, 1, 4, 4)}
    pseudo = {
        "p_soft": torch.full((1, 1, 4, 4), 0.8),
        "confidence": torch.ones(1, 1, 4, 4),
    }
    with pytest.raises(RuntimeError, match="must skip pseudo refiner"):
        encoder_pc_unlabeled_loss(outputs, aux, pseudo, _config(), 1)
