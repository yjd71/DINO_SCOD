import torch

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.PC_HBM.training.losses import (
    _quality_loss,
    base_structure_loss,
    decoder_base_loss,
    pc_hbm_labeled_loss,
    pc_hbm_pc_only_labeled_loss,
    pc_injection_strength,
    pc_mode_for_epoch,
    structure_loss,
)
from Model.PC_HBM.training.optimizer import migration_aware_parameter_groups


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
            "valid2_map": torch.ones(1, 28, 28, dtype=torch.bool),
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


def test_explicitly_reused_pc_parameters_use_half_learning_rate():
    module = torch.nn.Module()
    module.base = torch.nn.Linear(2, 2)
    module.pc_hbm = torch.nn.Linear(2, 2)
    groups = migration_aware_parameter_groups(
        module,
        base_lr=2.0e-4,
        reused_parameter_names=("base.weight", "pc_hbm.weight"),
    )
    by_name = {group["group_name"]: group for group in groups}
    assert by_name["base"]["lr"] == 2.0e-4
    assert by_name["migrated_pc"]["lr"] == 1.0e-4
    assert any(
        parameter is module.pc_hbm.weight
        for parameter in by_name["migrated_pc"]["params"]
    )
    assert any(
        parameter is module.base.weight for parameter in by_name["base"]["params"]
    )


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


def test_refined_boundary_gradient_is_limited_to_valid2_map():
    cfg = DinoPCHBMConfig()
    gt = torch.zeros(1, 1, 32, 32)
    outputs = _outputs()
    aux = _full_aux(outputs)
    refined = aux["p2_bra"]["B2_refined_map"]
    valid2 = torch.zeros(1, 28, 28, dtype=torch.bool)
    valid2[:, 0, 0] = True
    aux["p2_bra"]["valid2_map"] = valid2

    loss, _ = pc_hbm_labeled_loss(outputs, aux, gt, 11, cfg)
    loss.backward()

    assert refined.grad is not None
    assert refined.grad[0, 0, 0, 0].abs() > 0
    assert torch.count_nonzero(refined.grad[0, 0, 1:, :]) == 0
    assert torch.count_nonzero(refined.grad[0, 0, 0, 1:]) == 0


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


def test_decoder_base_loss_keeps_legacy_five_output_contract():
    cfg = DinoPCHBMConfig()
    outputs = _outputs()
    gt = torch.rand(1, 1, 16, 16)

    actual = decoder_base_loss(
        outputs,
        {"decoder_architecture": "legacy_transformer"},
        gt,
        cfg,
    )

    torch.testing.assert_close(actual, base_structure_loss(outputs, gt))


def test_decoder_base_loss_supervises_all_bgfbr_foreground_and_background_logits():
    cfg = DinoPCHBMConfig()
    cfg.lambda_bgfbr_fg = 1.5
    cfg.lambda_bgfbr_bg = 1.25
    cfg.lambda_bgfbr_final = 0.75
    cfg.lambda_bgfbr_global = 0.5
    outputs = _outputs()
    gt = (torch.rand(1, 1, 32, 32) > 0.5).float()
    target_output_scale = torch.nn.functional.interpolate(
        gt, size=(16, 16), mode="nearest"
    )
    foreground = tuple(
        torch.randn(1, 1, 16, 16, requires_grad=True) for _ in range(4)
    )
    background = tuple(
        torch.randn(1, 1, 16, 16, requires_grad=True) for _ in range(4)
    )
    aux = {
        "decoder_architecture": "bgfbr_pc_v1",
        "bgfbr": {"fg_output": foreground, "bg_output": background},
    }

    actual = decoder_base_loss(outputs, aux, gt, cfg)
    weights = (1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, 1.0 / 2.0)
    expected_fg = sum(
        weight * structure_loss(logit, target_output_scale)
        for weight, logit in zip(weights, foreground)
    )
    expected_bg = sum(
        weight * structure_loss(logit, 1.0 - target_output_scale)
        for weight, logit in zip(weights, background)
    )
    expected = (
        1.5 * expected_fg
        + 1.25 * expected_bg
        + 0.75 * structure_loss(outputs[3], target_output_scale)
        + 0.5 * structure_loss(outputs[4], target_output_scale)
    )
    torch.testing.assert_close(actual, expected)

    actual.backward()
    assert all(
        logit.grad is not None and torch.isfinite(logit.grad).all()
        for logit in foreground
    )
    assert all(
        logit.grad is not None and torch.isfinite(logit.grad).all()
        for logit in background
    )


def test_teacher_only_pc_loss_excludes_bgfbr_base_supervision():
    cfg = DinoPCHBMConfig()
    outputs = _outputs()
    gt = torch.rand(1, 1, 16, 16)
    aux = _full_aux(outputs)
    foreground = tuple(
        torch.randn(1, 1, 16, 16, requires_grad=True) for _ in range(4)
    )
    background = tuple(
        torch.randn(1, 1, 16, 16, requires_grad=True) for _ in range(4)
    )
    aux.update(
        {
            "decoder_architecture": "bgfbr_pc_v1",
            "bgfbr": {"fg_output": foreground, "bg_output": background},
        }
    )

    loss, log = pc_hbm_pc_only_labeled_loss(
        outputs,
        aux,
        gt,
        6,
        cfg,
        pc_mode="parent_only",
    )
    loss.backward()

    assert "L_base" not in log
    assert all(logit.grad is None for logit in foreground + background)
    assert all(
        output.grad is None or torch.count_nonzero(output.grad) == 0
        for output in outputs
    )
