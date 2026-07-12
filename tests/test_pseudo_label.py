import math

import pytest
import torch

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.PC_HBM.training.pseudo_label import (
    build_pc_confidence,
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


def test_hard_targets_keep_reliable_foreground_and_background():
    cfg = DinoPCHBMConfig()
    p = torch.tensor([[[[0.1, 0.4, 0.8]]]])
    pi = torch.tensor([0.97, 0.01, 0.01, 0.01]).view(1, 4, 1, 1).expand(1, 4, 1, 3)
    aux = {
        "p_final": p,
        "z_main": torch.logit(p.clamp(1e-4, 1 - 1e-4)),
        "pc_hbm": {"C23_map": torch.zeros_like(p), "route_entropy_norm": torch.zeros(1)},
        "mixture": {"pi": pi},
    }
    targets = prepare_pseudo_targets(aux, cfg)
    assert targets["hard_valid"].tolist() == [[[[True, False, True]]]]
    assert targets["hard_target"].tolist() == [[[[0.0, 0.0, 1.0]]]]
    assert targets["hard_weight"][0, 0, 0, 0] > 0


def test_weighted_loss_learns_from_all_background_target():
    logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
    target = torch.zeros_like(logits)
    confidence = torch.ones_like(logits)
    loss = weighted_structure_loss(logits, target, confidence)
    loss.backward()
    assert loss > 0
    assert logits.grad is not None and logits.grad.abs().sum() > 0


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
    assert log["hard_valid_ratio"] >= 0


def test_unlabeled_loss_rejects_same_shape_different_main_logit():
    cfg = DinoPCHBMConfig()
    z_main = torch.randn(1, 1, 8, 8, requires_grad=True)
    impostor = z_main.detach().clone().requires_grad_(True)
    outputs = (z_main, z_main, z_main, impostor, z_main)
    aux = {"z_main": z_main, "mixture_skipped": True}
    pseudo = torch.rand_like(z_main)
    confidence = torch.ones_like(z_main)

    with pytest.raises(ValueError, match="identify the Student main logit"):
        pc_unlabeled_loss(outputs, aux, pseudo, confidence, 1, cfg)
