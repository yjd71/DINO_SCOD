import math

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


def test_confidence_is_exact_five_factor_product_without_double_sigmoid():
    p_final = torch.full((1, 1, 2, 2), 0.9)
    p_main = 0.8
    pi = torch.tensor([0.7, 0.1, 0.1, 0.1]).view(1, 4, 1, 1).expand(1, 4, 2, 2)
    aux = {
        "p_final": p_final,
        "z_main": torch.full_like(p_final, torch.logit(torch.tensor(p_main))),
        "pc_hbm": {
            "C23_map": torch.full_like(p_final, 0.2),
            "route_entropy_norm": torch.tensor([0.25]),
        },
        "mixture": {"pi": pi},
    }
    confidence = build_pc_confidence(aux)
    entropy = -sum(value * math.log(value) for value in (0.7, 0.1, 0.1, 0.1)) / math.log(4)
    expected = (2 * abs(0.9 - 0.5)) * (1 - abs(0.9 - p_main)) * 0.8 * (1 - entropy) * 0.75
    torch.testing.assert_close(confidence, torch.full_like(confidence, expected))


def test_pseudo_targets_restore_hard_contract_and_clone_corrected_features():
    cfg = DinoPCHBMConfig()
    p = torch.tensor([[[[0.1, 0.4, 0.8]]]])
    pi = torch.tensor([0.97, 0.01, 0.01, 0.01]).view(1, 4, 1, 1).expand(1, 4, 1, 3)
    aux = {
        "p_final": p,
        "z_main": torch.logit(p.clamp(1e-4, 1 - 1e-4)),
        "pc_hbm": {"C23_map": torch.zeros_like(p), "route_entropy_norm": torch.zeros(1)},
        "mixture": {"pi": pi},
        "distill_features": {
            "p3_corr": torch.randn(1, 4, 2, 2),
            "p2_refined": torch.randn(1, 4, 2, 2),
        },
    }
    targets = prepare_pseudo_targets(aux, cfg)
    assert set(targets) == {
        "p_soft",
        "confidence",
        "hard_target",
        "hard_valid",
        "hard_weight",
        "distill_features",
    }
    torch.testing.assert_close(
        targets["hard_target"],
        torch.tensor([[[[0.0, 0.0, 1.0]]]]),
    )
    torch.testing.assert_close(
        targets["hard_valid"],
        torch.tensor([[[[True, False, True]]]]),
    )
    torch.testing.assert_close(
        targets["hard_weight"],
        targets["confidence"] * targets["hard_valid"],
    )
    for name in ("p3_corr", "p2_refined"):
        cloned = targets["distill_features"][name]
        original = aux["distill_features"][name]
        torch.testing.assert_close(cloned, original)
        assert cloned.data_ptr() != original.data_ptr()


def test_feature_distillation_is_confidence_weighted_and_student_only():
    student = torch.randn(2, 8, 4, 4, requires_grad=True)
    teacher = torch.randn(2, 8, 4, 4, requires_grad=True)
    confidence = torch.ones(2, 1, 8, 8)
    loss = confidence_weighted_feature_cosine_loss(student, teacher, confidence)
    loss.backward()
    assert torch.isfinite(loss)
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert teacher.grad is None

    zero_student = torch.randn(1, 4, 3, 3, requires_grad=True)
    zero_loss = confidence_weighted_feature_cosine_loss(
        zero_student,
        torch.randn_like(zero_student),
        torch.zeros(1, 1, 6, 6),
    )
    zero_loss.backward()
    assert zero_loss == 0
    assert zero_student.grad is not None and zero_student.grad.abs().sum() == 0


def test_weighted_loss_learns_from_all_background_target():
    logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
    target = torch.zeros_like(logits)
    confidence = torch.ones_like(logits)
    loss = weighted_structure_loss(logits, target, confidence)
    loss.backward()
    assert loss > 0
    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_weighted_hard_loss_is_differentiable_zero_for_an_empty_mask():
    logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
    target = torch.ones_like(logits)
    loss = weighted_structure_loss(logits, target, torch.zeros_like(logits))

    assert loss.requires_grad
    assert loss == 0
    loss.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() == 0


def test_hard_mask_gradient_keeps_reliable_foreground_and_background_only():
    logits = torch.zeros(1, 1, 1, 3, requires_grad=True)
    hard_target = torch.tensor([[[[0.0, 1.0, 1.0]]]])
    hard_weight = torch.tensor([[[[1.0, 0.0, 1.0]]]])

    loss = weighted_structure_loss(logits, hard_target, hard_weight)
    loss.backward()

    gradient = logits.grad[0, 0, 0]
    assert gradient[0] > 0  # reliable background
    assert gradient[1] == 0  # uncertain pixel
    assert gradient[2] < 0  # reliable foreground


def test_unlabeled_main_never_uses_z_final():
    cfg = DinoPCHBMConfig()
    tensors = [torch.randn(1, 1, 8, 8, requires_grad=True) for _ in range(5)]
    z_final = torch.randn(1, 1, 8, 8, requires_grad=True)
    pseudo = torch.rand(1, 1, 8, 8)
    confidence = torch.ones_like(pseudo)
    aux = {"z_main": tensors[3], "z_final": z_final, "mixture_skipped": True}
    loss, log = pc_unlabeled_loss(tensors, aux, pseudo, confidence, 1, cfg)
    loss.backward()
    assert tensors[3].grad is not None
    assert z_final.grad is None
    assert {
        "L_u_hard",
        "L_u_hard_weighted",
        "hard_ramp",
        "hard_valid_ratio",
    } <= set(log)
    assert log["pseudo_conf_positive_fraction"] == 1


@pytest.mark.parametrize(
    ("epoch", "expected_ramp"),
    ((1, 1 / 3), (2, 2 / 3), (3, 1.0), (8, 1.0)),
)
def test_hard_loss_matches_legacy_ramp_and_keeps_reliable_background(
    epoch, expected_ramp
):
    cfg = DinoPCHBMConfig()
    tensors = [torch.zeros(1, 1, 8, 8, requires_grad=True) for _ in range(5)]
    pseudo = torch.full((1, 1, 8, 8), 0.1)
    confidence = torch.ones_like(pseudo)
    aux = {"z_main": tensors[3], "mixture_skipped": True}

    loss, log = pc_unlabeled_loss(tensors, aux, pseudo, confidence, epoch, cfg)
    expected = (
        log["L_u_soft"]
        + log["L_u_side"]
        + cfg.hard_loss_weight * expected_ramp * log["L_u_hard"]
    )
    torch.testing.assert_close(loss.detach(), expected)
    torch.testing.assert_close(log["hard_ramp"], torch.tensor(expected_ramp))
    torch.testing.assert_close(log["hard_valid_ratio"], torch.tensor(1.0))
    assert log["L_u_hard"] > 0

    loss.backward()
    assert tensors[3].grad is not None and tensors[3].grad.abs().sum() > 0


def test_hard_loss_is_differentiable_zero_when_no_pseudo_pixel_is_reliable():
    cfg = DinoPCHBMConfig()
    tensors = [torch.zeros(1, 1, 8, 8, requires_grad=True) for _ in range(5)]
    pseudo = torch.full((1, 1, 8, 8), 0.5)
    confidence = torch.ones_like(pseudo)
    aux = {"z_main": tensors[3], "mixture_skipped": True}

    loss, log = pc_unlabeled_loss(tensors, aux, pseudo, confidence, 1, cfg)
    assert log["L_u_hard"] == 0
    assert log["L_u_hard_weighted"] == 0
    assert log["hard_valid_ratio"] == 0
    loss.backward()
    assert tensors[3].grad is not None and torch.isfinite(tensors[3].grad).all()


def test_hard_pseudo_can_be_disabled_for_ablation():
    cfg = DinoPCHBMConfig(use_hard_pseudo=False)
    tensors = [torch.zeros(1, 1, 8, 8, requires_grad=True) for _ in range(5)]
    pseudo = torch.full((1, 1, 8, 8), 0.9)
    confidence = torch.ones_like(pseudo)
    aux = {"z_main": tensors[3], "mixture_skipped": True}

    loss, log = pc_unlabeled_loss(tensors, aux, pseudo, confidence, 1, cfg)
    torch.testing.assert_close(loss.detach(), log["L_u_soft"] + log["L_u_side"])
    assert log["L_u_hard"] == 0
    assert log["hard_ramp"] == 0


def test_unlabeled_total_includes_hard_and_feature_distillation_before_lambda_u():
    cfg = DinoPCHBMConfig(lambda_u=0.37)
    outputs = [torch.randn(1, 1, 8, 8, requires_grad=True) for _ in range(5)]
    student_p3 = torch.randn(1, 4, 4, 4, requires_grad=True)
    student_p2 = torch.randn(1, 4, 4, 4, requires_grad=True)
    aux = {
        "z_main": outputs[3],
        "mixture_skipped": True,
        "features": {"p3": student_p3, "p2": student_p2},
    }
    teacher_features = {
        "p3_corr": torch.randn_like(student_p3),
        "p2_refined": torch.randn_like(student_p2),
    }
    pseudo = torch.full((1, 1, 8, 8), 0.9)
    confidence = torch.ones_like(pseudo)

    loss, log = pc_unlabeled_loss(
        outputs,
        aux,
        pseudo,
        confidence,
        2,
        cfg,
        teacher_features=teacher_features,
    )
    expected = cfg.lambda_u * (
        log["L_u_soft"]
        + log["L_u_hard_weighted"]
        + log["L_u_side"]
        + log["L_u_feature"]
    )
    torch.testing.assert_close(loss.detach(), expected)
    torch.testing.assert_close(log["loss_unlabeled"], loss.detach())
    assert log["L_u_hard"] > 0
    assert log["L_u_feature"] > 0

    loss.backward()
    assert outputs[3].grad is not None and outputs[3].grad.abs().sum() > 0
    assert student_p3.grad is not None and student_p3.grad.abs().sum() > 0
    assert student_p2.grad is not None and student_p2.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "kwargs",
    (
        {"hard_loss_weight": -1.0},
        {"pseudo_hard_ramp_epochs": 0},
        {"pseudo_bg_threshold": 0.5},
        {"pseudo_fg_threshold": 0.5},
    ),
)
def test_hard_pseudo_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        DinoPCHBMConfig(**kwargs)


def test_unlabeled_loss_accepts_ddp_cloned_main_logit():
    cfg = DinoPCHBMConfig()
    source = torch.randn(1, 1, 8, 8, requires_grad=True)
    output_z_main = source.clone()
    aux_z_main = source.clone()
    aux_z_main.retain_grad()
    sides = [torch.randn_like(source, requires_grad=True) for _ in range(4)]
    outputs = (sides[0], sides[1], sides[2], output_z_main, sides[3])
    aux = {"z_main": aux_z_main, "mixture_skipped": True}
    pseudo = torch.rand_like(source)
    confidence = torch.ones_like(source)

    assert output_z_main.data_ptr() != aux_z_main.data_ptr()
    loss, _ = pc_unlabeled_loss(outputs, aux, pseudo, confidence, 1, cfg)
    loss.backward()

    assert aux_z_main.grad is not None and torch.isfinite(aux_z_main.grad).all()
    assert source.grad is not None and torch.isfinite(source.grad).all()


@pytest.mark.parametrize(
    ("output_z_main", "match"),
    [
        (torch.randn(1, 1, 7, 8), "shape, device, and dtype"),
        (torch.randn(1, 1, 8, 8, dtype=torch.float64), "shape, device, and dtype"),
    ],
)
def test_unlabeled_loss_rejects_incompatible_main_logit_contract(output_z_main, match):
    cfg = DinoPCHBMConfig()
    z_main = torch.randn(1, 1, 8, 8, requires_grad=True)
    outputs = (z_main, z_main, z_main, output_z_main, z_main)
    aux = {"z_main": z_main, "mixture_skipped": True}
    pseudo = torch.rand_like(z_main)
    confidence = torch.ones_like(z_main)

    with pytest.raises(ValueError, match=match):
        pc_unlabeled_loss(outputs, aux, pseudo, confidence, 1, cfg)
