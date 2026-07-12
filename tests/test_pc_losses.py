import torch

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.PC_HBM.training.losses import (
    _quality_loss,
    base_structure_loss,
    pc_hbm_labeled_loss,
    pc_injection_strength,
    pc_mode_for_epoch,
)


def _outputs(size=16):
    return tuple(torch.randn(1, 1, size, size, requires_grad=True) for _ in range(5))


def _full_aux(outputs, size=16):
    indices = {
        "batch_ids": torch.tensor([0, 0]),
        "flat_indices": torch.tensor([0, 1]),
    }
    pc = {
        "P3_group": torch.softmax(torch.randn(2, 4, requires_grad=True), dim=1),
        "boundary_indices3": indices,
        "top_parent_region_ids": torch.tensor([[0, 1], [2, 3]]),
        "top_parent_valid": torch.ones(2, 2, dtype=torch.bool),
        "S_child": torch.randn(2, 2, requires_grad=True),
        "G_attn": torch.randn(2, 6, requires_grad=True),
        "O_pc_token": torch.randn(2, 2, requires_grad=True),
        "gate_pc_token": torch.full((2, 1), 0.4, requires_grad=True),
        "C23_token": torch.full((2, 1), 0.2),
        "B3": torch.full((1, 1, 28, 28), 0.5, requires_grad=True),
        "C23_map": torch.full((1, 1, 28, 28), 0.2),
        "gate_pc_map": torch.full((1, 1, 28, 28), 0.4),
        "route_entropy_norm": torch.tensor([0.2]),
    }
    branches = {
        name: torch.randn(1, 1, size, size, requires_grad=True)
        for name in ("z_keep", "z_res", "z_def", "z_sup")
    }
    mix_logits = torch.randn(1, 4, size, size, requires_grad=True)
    mixture = {
        **branches,
        "pi": torch.softmax(mix_logits, dim=1),
        "branch_quality": torch.randn(1, 4, size, size, requires_grad=True),
        "B_pix": torch.ones(1, 1, size, size),
        "O_pix": torch.randn(1, 2, size, size, requires_grad=True),
        "Mask_corr": torch.sigmoid(torch.randn(1, 1, size, size, requires_grad=True)),
    }
    mixture["z_final"] = sum(
        mixture["pi"][:, i : i + 1] * mixture[name]
        for i, name in enumerate(("z_keep", "z_res", "z_def", "z_sup"))
    )
    return {
        "pc_active": True,
        "forward_mode": "full",
        "fallback_reason": None,
        "z_main": outputs[3],
        "z_final": mixture["z_final"],
        "pc_hbm": pc,
        "p2_bra": {
            "B2": torch.full((1, 1, 28, 28), 0.5, requires_grad=True),
            "B2_refined_map": torch.full((1, 1, 28, 28), 0.5, requires_grad=True),
        },
        "p1_pra": {"B1": torch.full((1, 1, size, size), 0.5, requires_grad=True)},
        "mixture": mixture,
    }


def test_schedule_and_ramp_are_one_based():
    cfg = DinoPCHBMConfig()
    assert [pc_mode_for_epoch(e, cfg) for e in (1, 5, 6, 10, 11)] == [
        "off",
        "off",
        "parent_only",
        "parent_only",
        "full",
    ]
    assert [pc_injection_strength(e, cfg) for e in (10, 11, 12, 13)] == [0.0, 1 / 3, 2 / 3, 1.0]


def test_off_and_parent_only_loss_matrix():
    cfg = DinoPCHBMConfig()
    gt = (torch.rand(1, 1, 32, 32) > 0.5).float()
    outputs = _outputs()
    off, off_log = pc_hbm_labeled_loss(outputs, None, gt, 1, cfg, pc_mode="off")
    torch.testing.assert_close(off, base_structure_loss(outputs, gt))
    assert off_log["L_final"] == 0

    aux = _full_aux(outputs)
    aux["forward_mode"] = "parent_only"
    parent, log = pc_hbm_labeled_loss(outputs, aux, gt, 6, cfg, pc_mode="parent_only")
    expected = base_structure_loss(outputs, gt) + 0.2 * log["L_parent"] + 0.1 * log["L_B3"]
    torch.testing.assert_close(parent.detach(), expected)
    assert log["L_child"] == 0 and log["L_final"] == 0


def test_full_loss_is_finite_and_backpropagates():
    cfg = DinoPCHBMConfig()
    gt = (torch.rand(1, 1, 32, 32) > 0.5).float()
    outputs = _outputs()
    aux = _full_aux(outputs)
    loss, log = pc_hbm_labeled_loss(outputs, aux, gt, 11, cfg)
    assert torch.isfinite(loss)
    assert abs(log["pc_strength"].item() - 1 / 3) < 1.0e-6
    loss.backward()
    assert outputs[3].grad is not None
    assert aux["pc_hbm"]["S_child"].grad is not None


def test_quality_loss_normalizes_across_all_four_branches():
    reference = torch.zeros(1, 1, 2, 2, requires_grad=True)
    mixture = {
        "branch_quality": torch.zeros(1, 4, 2, 2, requires_grad=True),
        "B_pix": torch.ones(1, 1, 2, 2),
    }
    pixel_error = torch.cat(
        (torch.ones(1, 1, 2, 2), torch.zeros(1, 3, 2, 2)), dim=1
    )
    loss = _quality_loss(mixture, {"pixel_error": pixel_error}, reference)
    torch.testing.assert_close(loss, torch.tensor(0.375))


def test_training_fails_fast_on_memory_fallback():
    cfg = DinoPCHBMConfig()
    outputs = _outputs()
    gt = torch.zeros(1, 1, 16, 16)
    aux = {"pc_active": False, "fallback_reason": "memory_not_ready"}
    try:
        pc_hbm_labeled_loss(outputs, aux, gt, 11, cfg)
    except RuntimeError as error:
        assert "fallback" in str(error)
    else:
        raise AssertionError("full PC-HBM training must reject baseline fallback")
