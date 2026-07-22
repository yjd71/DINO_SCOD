from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from configs.pc_hbm_dino_config import EncoderPCHBMConfig
from Model.PC_HBM.encoder.encoder_pc_adapter import EncoderPCHBMAdapter
from Model.PC_HBM.training import (
    EncoderPCStage,
    build_encoder_pc_optimizer,
    configure_encoder_pc_stage,
    encoder_pc_labeled_loss,
    make_ema_encoder_adapter,
    update_ema_encoder_adapter,
)
from Model.PC_HBM.training.supervision import (
    build_gt_boundary,
    build_region_label_map,
)


class _ModelWithFrozenDino(nn.Module):
    def __init__(self, adapter: EncoderPCHBMAdapter) -> None:
        super().__init__()
        self.dino = nn.Linear(4, 4)
        self.encoder_pc_hbm = adapter


def _small_head() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 1))


def _has_trainable(module: nn.Module) -> bool:
    return any(parameter.requires_grad for parameter in module.parameters())


@pytest.mark.parametrize(
    ("epoch", "index", "name", "mode"),
    (
        (1, 1, "bootstrap", "bootstrap"),
        (5, 1, "bootstrap", "bootstrap"),
        (6, 2, "parent_only", "parent_only"),
        (10, 2, "parent_only", "parent_only"),
        (11, 3, "parent_child_f3", "full"),
        (15, 3, "parent_child_f3", "full"),
        (16, 4, "hierarchical_full", "full"),
        (20, 4, "hierarchical_full", "full"),
        (21, 5, "hierarchical_refiner", "full"),
        (30, 5, "hierarchical_refiner", "full"),
    ),
)
def test_stage_boundaries_are_stable(
    epoch: int,
    index: int,
    name: str,
    mode: str,
) -> None:
    stage = EncoderPCStage.for_epoch(epoch, EncoderPCHBMConfig())

    assert stage.epoch == epoch
    assert stage.index == index
    assert stage.name == name
    assert stage.mode == mode
    assert stage.enable_refiner is (epoch >= 21)


def test_stage_flags_keep_route_supervision_and_curriculum_separate() -> None:
    config = EncoderPCHBMConfig()
    bootstrap = EncoderPCStage.for_epoch(5, config)
    parent = EncoderPCStage.for_epoch(6, config)
    f4_f3 = EncoderPCStage.for_epoch(11, config)
    f2_f1 = EncoderPCStage.for_epoch(16, config)

    bootstrap_flags = bootstrap.adapter_flags()
    parent_flags = parent.adapter_flags()
    f4_f3_flags = f4_f3.adapter_flags()
    f2_f1_flags = f2_f1.adapter_flags()

    assert not bootstrap_flags.require_same_image_positive
    assert parent_flags.require_same_image_positive
    assert not parent_flags.enable_f4_f3
    assert f4_f3_flags.enable_f4_f3
    assert f4_f3_flags.f4_f3_progress == pytest.approx(1.0 / 5.0)
    assert not f4_f3_flags.enable_f2_f1
    assert f2_f1_flags.enable_f2_f1
    assert f2_f1_flags.f2_f1_progress == pytest.approx(1.0 / 5.0)
    assert not parent.adapter_flags(False).require_same_image_positive

    with pytest.raises(ValueError):
        EncoderPCStage.for_epoch(0, config)
    with pytest.raises(ValueError):
        EncoderPCStage.for_epoch(31, config)


def test_configure_stage_applies_exact_gradient_ownership() -> None:
    config = EncoderPCHBMConfig()
    adapter = EncoderPCHBMAdapter(config)
    decoder = _small_head()
    refiner = _small_head()

    configure_encoder_pc_stage(
        adapter, decoder, refiner, EncoderPCStage.for_epoch(1, config)
    )
    assert _has_trainable(adapter.bootstrap)
    assert not any(
        _has_trainable(projector)
        for projector in adapter.bootstrap.projector.cls_projectors
    )
    assert not _has_trainable(adapter.router)
    assert not _has_trainable(adapter.verifier)
    assert not _has_trainable(adapter.route_context)
    assert not _has_trainable(adapter.injector)
    assert not _has_trainable(adapter.propagation)
    assert _has_trainable(decoder)
    assert not _has_trainable(refiner)

    configure_encoder_pc_stage(
        adapter, decoder, refiner, EncoderPCStage.for_epoch(6, config)
    )
    assert _has_trainable(adapter.router)
    assert not _has_trainable(adapter.verifier)
    assert [
        _has_trainable(projector)
        for projector in adapter.bootstrap.projector.cls_projectors
    ] == [False, False, False, True]

    configure_encoder_pc_stage(
        adapter, decoder, refiner, EncoderPCStage.for_epoch(11, config)
    )
    assert _has_trainable(adapter.verifier)
    assert _has_trainable(adapter.route_context)
    assert _has_trainable(adapter.injector)
    assert not _has_trainable(adapter.propagation)

    configure_encoder_pc_stage(
        adapter, decoder, refiner, EncoderPCStage.for_epoch(20, config)
    )
    assert _has_trainable(adapter.propagation)
    assert not _has_trainable(refiner)

    configure_encoder_pc_stage(
        adapter, decoder, refiner, EncoderPCStage.for_epoch(21, config)
    )
    assert _has_trainable(adapter.propagation)
    assert _has_trainable(refiner)


def test_optimizer_retains_inactive_parameters_and_excludes_dino() -> None:
    config = EncoderPCHBMConfig()
    adapter = EncoderPCHBMAdapter(config)
    model = _ModelWithFrozenDino(adapter)
    decoder = _small_head()
    refiner = _small_head()
    configure_encoder_pc_stage(
        model, decoder, refiner, EncoderPCStage.for_epoch(1, config)
    )

    optimizer = build_encoder_pc_optimizer(
        model,
        decoder,
        refiner,
        decoder_warm_started=True,
    )
    by_name = {group["name"]: group for group in optimizer.param_groups}

    assert set(by_name) == {"decoder", "encoder_pc_hbm", "pseudo_refiner"}
    assert by_name["decoder"]["lr"] == pytest.approx(3.0e-5)
    assert by_name["encoder_pc_hbm"]["lr"] == pytest.approx(1.0e-4)
    assert by_name["pseudo_refiner"]["lr"] == pytest.approx(1.0e-4)
    assert all(group["weight_decay"] == 0.0 for group in optimizer.param_groups)

    optimized_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected_ids = {
        id(parameter)
        for module in (adapter, decoder, refiner)
        for parameter in module.parameters()
    }
    dino_ids = {id(parameter) for parameter in model.dino.parameters()}
    frozen_future_ids = {
        id(parameter)
        for module in (adapter.verifier, adapter.injector, adapter.propagation)
        for parameter in module.parameters()
    }
    assert optimized_ids == expected_ids
    assert optimized_ids.isdisjoint(dino_ids)
    assert frozen_future_ids <= optimized_ids
    assert all(not parameter.requires_grad for parameter in adapter.verifier.parameters())

    scratch = build_encoder_pc_optimizer(adapter, _small_head())
    assert scratch.param_groups[0]["lr"] == pytest.approx(1.0e-4)


def test_memory_adapter_ema_updates_parameters_and_copies_buffers() -> None:
    adapter = EncoderPCHBMAdapter(EncoderPCHBMConfig())
    adapter.register_buffer("ema_contract_buffer", torch.tensor([1.0]))
    ema = make_ema_encoder_adapter(adapter)
    name, online_parameter = next(iter(adapter.named_parameters()))
    ema_parameter = dict(ema.named_parameters())[name]
    old_ema = ema_parameter.detach().clone()

    with torch.no_grad():
        online_parameter.add_(2.0)
        adapter.ema_contract_buffer.fill_(7.0)
    update_ema_encoder_adapter(ema, adapter, decay=0.5)

    assert torch.allclose(ema_parameter, old_ema + 1.0)
    assert torch.equal(ema.ema_contract_buffer, adapter.ema_contract_buffer)
    assert not ema.training
    assert not any(parameter.requires_grad for parameter in ema.parameters())


def _labeled_case() -> tuple[
    list[torch.Tensor],
    dict[str, object],
    torch.Tensor,
    dict[str, torch.Tensor],
]:
    gt = torch.zeros(1, 1, 16, 16)
    gt[:, :, 4:12, 4:12] = 1.0
    outputs = [
        torch.zeros(1, 1, 16, 16, requires_grad=True) for _ in range(5)
    ]
    coarse_logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
    boundary_logits = torch.zeros(1, 1, 4, 4, requires_grad=True)
    route_info_nce = torch.tensor(2.0, requires_grad=True)
    batch_ids = torch.tensor([0, 0], dtype=torch.long)
    flat_indices = torch.tensor([5, 6], dtype=torch.long)
    region_map = build_region_label_map(gt, size=(4, 4))
    query_regions = region_map[batch_ids, flat_indices // 4, flat_indices % 4]
    candidate_regions = torch.stack(
        (
            query_regions,
            (query_regions + 1) % 4,
            (query_regions + 2) % 4,
        ),
        dim=1,
    )
    parent_values = torch.zeros(2, 3, 8)
    parent_values.scatter_(2, candidate_regions.unsqueeze(-1), 1.0)
    parent_scores = torch.tensor(
        [[1.0, 0.0, -1.0], [1.0, 0.0, -1.0]], requires_grad=True
    )
    parent_valid = torch.ones(2, 3, dtype=torch.bool)
    semantic_logits = torch.zeros(2, 3, requires_grad=True)
    detail_logits = torch.zeros(2, 3, requires_grad=True)
    query_geometry = torch.ones(2, 6, requires_grad=True)
    gate_logits = torch.full(
        (1, 1, 4, 4), math.log(4.0), requires_grad=True
    )
    f4_delta = (
        torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) / 16.0
    ).requires_grad_()
    f3_delta = torch.ones(1, 1, 4, 4, requires_grad=True)
    f2_delta = torch.full((1, 1, 4, 4), 2.0, requires_grad=True)
    f1_delta = torch.full((1, 1, 4, 4), 3.0, requires_grad=True)

    encoder_aux: dict[str, object] = {
        "coarse_logits": coarse_logits,
        "coarse_probability": coarse_logits.sigmoid(),
        "boundary_logits": boundary_logits,
        "boundary_probability": boundary_logits.sigmoid(),
        "boundary_indices": {
            "batch_ids": batch_ids,
            "flat_indices": flat_indices,
        },
        "route": {"route_info_nce": route_info_nce},
        "parent": {
            "top_parent_values": parent_values,
            "top_parent_valid": parent_valid,
            "top_parent_scores": parent_scores,
            "parent_attention": parent_scores.softmax(dim=1),
        },
        "verification": {
            "S_semantic": semantic_logits,
            "S_detail": detail_logits,
            "top_parent_region_ids": candidate_regions,
            "top_parent_valid": parent_valid,
            "query_valid": torch.ones(2, dtype=torch.bool),
        },
        "query_geometry": query_geometry,
        "route_context": {
            "gate_map": gate_logits.sigmoid(),
            "valid3_map": torch.ones(1, 1, 4, 4, dtype=torch.bool),
            "c23_map": torch.zeros(1, 1, 4, 4),
        },
        "C23_map": torch.zeros(1, 1, 4, 4),
        "injection": {"f4_delta": f4_delta, "f3_delta": f3_delta},
        "propagation": {"f2_delta": f2_delta, "f1_delta": f1_delta},
    }
    decoder_aux = {"decoder_architecture": "legacy_transformer"}
    leaves = {
        "coarse": coarse_logits,
        "boundary": boundary_logits,
        "route": route_info_nce,
        "parent": parent_scores,
        "semantic": semantic_logits,
        "detail": detail_logits,
        "geometry": query_geometry,
        "gate": gate_logits,
        "f4": f4_delta,
        "f3": f3_delta,
        "f2": f2_delta,
        "f1": f1_delta,
        "decoder": outputs[3],
    }
    return outputs, {"decoder": decoder_aux, "encoder": encoder_aux}, gt, leaves


def test_labeled_loss_stage_matrix_and_probability_contract() -> None:
    config = EncoderPCHBMConfig()
    outputs, aux, gt, _ = _labeled_case()

    _, bootstrap = encoder_pc_labeled_loss(
        outputs, aux, gt, config, EncoderPCStage.for_epoch(1, config)
    )
    assert bootstrap["L_encoder_coarse"].item() == pytest.approx(math.log(2.0))
    assert bootstrap["L_encoder_boundary"].item() == pytest.approx(math.log(2.0))
    assert bootstrap["L_route"].item() == 0.0
    assert bootstrap["L_parent"].item() == 0.0
    assert bootstrap["L_child_semantic"].item() == 0.0

    _, parent = encoder_pc_labeled_loss(
        outputs, aux, gt, config, EncoderPCStage.for_epoch(6, config)
    )
    assert parent["L_route"].item() == pytest.approx(2.0)
    assert parent["L_parent"].item() > 0.0
    assert parent["L_child_semantic"].item() == 0.0

    _, verification = encoder_pc_labeled_loss(
        outputs, aux, gt, config, EncoderPCStage.for_epoch(11, config)
    )
    assert verification["L_child_semantic"].item() > 0.0
    assert verification["L_child_detail"].item() > 0.0
    assert verification["L_geometry"].item() > 0.0
    boundary = build_gt_boundary(gt, size=(4, 4))

    def expected_injection(*deltas: torch.Tensor) -> float:
        magnitude = sum(delta.detach().abs().mean() for delta in deltas)
        stable = sum(
            ((1.0 - boundary) * delta.detach().abs().mean(dim=1, keepdim=True)).mean()
            for delta in deltas
        )
        smooth = sum(
            (delta.detach()[:, :, 1:, :] - delta.detach()[:, :, :-1, :]).abs().mean()
            + (delta.detach()[:, :, :, 1:] - delta.detach()[:, :, :, :-1]).abs().mean()
            for delta in deltas
        )
        return float(magnitude + 0.5 * stable + 0.25 * smooth)

    encoder_aux = aux["encoder"]
    injection = encoder_aux["injection"]
    propagation = encoder_aux["propagation"]
    expected_f4_f3 = expected_injection(
        injection["f4_delta"], injection["f3_delta"]
    )
    assert verification["L_injection"].item() == pytest.approx(expected_f4_f3)
    # gate_map is a sigmoid probability of 0.8 and every selected query needs
    # correction, so probability BCE is -log(0.8), not BCEWithLogits(0.8, 1).
    assert verification["L_gate"].item() == pytest.approx(-math.log(0.8), rel=1e-5)

    _, hierarchy = encoder_pc_labeled_loss(
        outputs, aux, gt, config, EncoderPCStage.for_epoch(16, config)
    )
    expected_hierarchy = expected_injection(
        injection["f4_delta"],
        injection["f3_delta"],
        propagation["f2_delta"],
        propagation["f1_delta"],
    )
    assert hierarchy["L_injection"].item() == pytest.approx(expected_hierarchy)
    assert hierarchy["refiner_enabled"].item() == 0.0

    _, refiner = encoder_pc_labeled_loss(
        outputs, aux, gt, config, EncoderPCStage.for_epoch(21, config)
    )
    assert refiner["refiner_enabled"].item() == 1.0


def test_route_stage_fails_fast_without_forward_info_nce() -> None:
    config = EncoderPCHBMConfig()
    outputs, aux, gt, _ = _labeled_case()
    del aux["encoder"]["route"]["route_info_nce"]

    with pytest.raises(RuntimeError, match="route_info_nce"):
        encoder_pc_labeled_loss(
            outputs, aux, gt, config, EncoderPCStage.for_epoch(6, config)
        )


def test_stage_loss_gradients_belong_only_to_active_modules() -> None:
    config = EncoderPCHBMConfig()
    outputs, aux, gt, leaves = _labeled_case()
    bootstrap_total, _ = encoder_pc_labeled_loss(
        outputs, aux, gt, config, EncoderPCStage.for_epoch(1, config)
    )
    bootstrap_total.backward()

    assert leaves["decoder"].grad is not None
    assert leaves["coarse"].grad is not None
    assert leaves["boundary"].grad is not None
    for name in ("route", "parent", "semantic", "detail", "geometry", "gate", "f4", "f3", "f2", "f1"):
        assert leaves[name].grad is None, name

    outputs, aux, gt, leaves = _labeled_case()
    verification_total, _ = encoder_pc_labeled_loss(
        outputs, aux, gt, config, EncoderPCStage.for_epoch(11, config)
    )
    verification_total.backward()

    for name in (
        "decoder",
        "coarse",
        "boundary",
        "route",
        "parent",
        "semantic",
        "detail",
        "geometry",
        "gate",
        "f4",
        "f3",
    ):
        assert leaves[name].grad is not None, name
    assert leaves["f2"].grad is None
    assert leaves["f1"].grad is None
