from __future__ import annotations

import pytest
import torch

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.decoder import Decoder
from Model.PC_HBM.memory import PCMemory
from Model.PC_HBM.training import pc_hbm_labeled_loss, prepare_pseudo_targets


def _ready_memory(config: DinoPCHBMConfig) -> PCMemory:
    image_ids = ("memory-a", "memory-b")
    count = 8
    route = {
        name: torch.randn(len(image_ids), config.memory_dim)
        for name in (
            "x3_global",
            "x3_boundary",
            "x3_uncertain",
            "x3_bg_near",
            "x3_environment",
        )
    }
    route["route_embed"] = torch.randn(len(image_ids), config.memory_dim)
    route["img_ids"] = list(image_ids)
    metadata = [
        {"image_id": image_ids[index % len(image_ids)], "region": "fg_boundary"}
        for index in range(count)
    ]
    memory = PCMemory(config.memory_dim, config.value_dim, config.geometry_dim)
    memory.append(
        {
            "source": "labeled_only",
            "route": route,
            "parent": {
                "p3_keys": torch.randn(count, config.memory_dim),
                "p3_values": torch.randn(count, config.value_dim),
                "p3_geometry": torch.randn(count, config.geometry_dim),
                "child_ptr": torch.arange(count),
                "parent_meta": metadata,
            },
            "child": {
                "p2_child_keys": torch.randn(count, config.memory_dim),
                "p2_child_geo": torch.randn(count, config.geometry_dim),
                "child_meta": metadata,
            },
        }
    )
    memory.finalize(
        compat_meta=config.expected_memory_meta(producer_fingerprint="unit-test")
    )
    return memory


@pytest.fixture(scope="module")
def decoder_inputs():
    torch.manual_seed(9)
    config = DinoPCHBMConfig(
        route_top_img_k=2,
        parent_topk=4,
        p3_min_tokens=8,
        p3_max_tokens=8,
        p2_min_tokens=8,
        p2_max_tokens=8,
        p1_min_tokens=16,
        p1_max_tokens=16,
        query_chunk_size=64,
    )
    model = Decoder(pc_cfg=config)
    features = [torch.randn(1, 28 * 28, 768) for _ in range(4)]
    return model, features, _ready_memory(config)


@pytest.mark.parametrize(
    ("mode", "pc_active", "mixture_skipped", "has_probability"),
    (
        ("off", False, True, True),
        ("parent_only", True, True, True),
        ("full", True, False, True),
        ("teacher_pseudo", True, False, True),
        ("student_core", True, True, False),
    ),
)
def test_decoder_pc_modes_have_stable_outputs_and_aux_schema(
    decoder_inputs, mode, pc_active, mixture_skipped, has_probability
):
    model, features, memory = decoder_inputs
    outputs, aux = model(
        features,
        memory=memory,
        pc_mode=mode,
        epoch=11,
        return_aux=True,
        query_image_ids=["query"],
    )
    assert len(outputs) == 5
    assert all(output.shape == (1, 1, 98, 98) for output in outputs)
    assert all(torch.isfinite(output).all() for output in outputs)
    assert aux["pc_active"] is pc_active
    assert aux["mixture_skipped"] is mixture_skipped
    assert (aux["p_final"] is not None) is has_probability
    if aux["p_final"] is not None:
        assert aux["p_final"].amin() >= 0
        assert aux["p_final"].amax() <= 1
    if mode in {"full", "teacher_pseudo"}:
        torch.testing.assert_close(aux["z_final"], outputs[3])
    if mode == "off":
        assert aux["features"]["p3"].shape == (1, 128, 28, 28)
        assert aux["features"]["p2"].shape == (1, 128, 28, 28)
    if mode == "teacher_pseudo":
        assert aux["distill_features"]["p3_corr"].shape == (1, 128, 28, 28)
        assert aux["distill_features"]["p2_refined"].shape == (1, 128, 28, 28)
    if mode == "student_core":
        assert aux["forward_mode"] == "student_core"
        assert aux["z_final"] is None
        assert aux["p_final"] is None
        assert aux["mixture"] is None
        assert aux["mixture_skipped"] is True
        assert "features" not in aux
        assert set(aux["pc_hbm"]) == {"p3_corr"}
        assert aux["pc_hbm"]["p3_corr"].shape == (1, 128, 28, 28)
        assert set(aux["p2_bra"]) == {"p2_refined"}
        assert aux["p2_bra"]["p2_refined"].shape == (1, 128, 28, 28)
        assert set(aux["p1_pra"]) == {
            "B1",
            "G1_raw_map",
            "R1_map",
            "O1_map",
            "R_sup_map",
            "valid1_map",
        }


def test_student_core_runs_p1_once_and_never_runs_mixture(
    decoder_inputs, monkeypatch
):
    model, features, memory = decoder_inputs
    calls = {"p1": 0, "mixture": 0}
    original_forward_p1 = model.pc_hbm.forward_p1
    original_forward_mixture = model.pc_hbm.forward_mixture

    def counted_forward_p1(*args, **kwargs):
        calls["p1"] += 1
        return original_forward_p1(*args, **kwargs)

    def counted_forward_mixture(*args, **kwargs):
        calls["mixture"] += 1
        return original_forward_mixture(*args, **kwargs)

    monkeypatch.setattr(model.pc_hbm, "forward_p1", counted_forward_p1)
    monkeypatch.setattr(model.pc_hbm, "forward_mixture", counted_forward_mixture)

    _, aux = model(
        features,
        memory=memory,
        pc_mode="student_core",
        epoch=11,
        return_aux=True,
        query_image_ids=["query"],
    )

    assert calls == {"p1": 1, "mixture": 0}
    assert aux["p1_pra"] is not None
    assert aux["mixture"] is None


def test_missing_or_incompatible_memory_returns_explicit_baseline_fallback(
    decoder_inputs,
):
    model, features, memory = decoder_inputs
    was_training = model.training
    model.eval()
    baseline = model(features)
    try:
        missing_outputs, missing_aux = model(
            features, pc_mode="full", return_aux=True
        )
        assert missing_aux["fallback_reason"] == "memory_missing"
        for actual, expected in zip(missing_outputs, baseline):
            torch.testing.assert_close(actual, expected)

        original_token_hw = memory.compat_meta["token_hw"]
        try:
            memory.compat_meta["token_hw"] = (14, 14)
            _, incompatible_aux = model(
                features, memory=memory, pc_mode="full", return_aux=True
            )
            assert incompatible_aux["fallback_reason"] == "compat_mismatch:token_hw"
        finally:
            memory.compat_meta["token_hw"] = original_token_hw
    finally:
        model.train(was_training)


def test_full_mode_backward_reaches_pc_modules(decoder_inputs):
    model, features, memory = decoder_inputs
    model.zero_grad(set_to_none=True)
    outputs, aux = model(
        features,
        memory=memory,
        pc_mode="full",
        epoch=11,
        return_aux=True,
        query_image_ids=["query"],
    )
    loss = sum(output.mean() for output in outputs) + aux["z_final"].mean()
    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.pc_hbm.parameters()
    )


def test_decoder_memory_feature_builder_produces_labeled_only_ready_memory(
    decoder_inputs,
):
    model, features, _ = decoder_inputs
    gt = torch.zeros(1, 1, 98, 98)
    gt[:, :, 20:78, 24:72] = 1
    memory_features = model.forward_memory_features(features)
    entries = model.pc_hbm.build_memory_entries(
        memory_features, gt, ["labeled-sample"]
    )
    memory = PCMemory(
        model.pc_cfg.memory_dim,
        model.pc_cfg.value_dim,
        model.pc_cfg.geometry_dim,
    )
    memory.append(entries)
    memory.finalize(
        compat_meta=model.pc_cfg.expected_memory_meta(
            producer_fingerprint="unit-test"
        )
    )
    assert memory.is_ready()
    assert set(memory.route["img_ids"]) == {"labeled-sample"}
    assert memory.parent["p3_keys"].device.type == "cpu"
    assert memory.parent["p3_keys"].dtype == torch.float16
    assert memory.parent["child_ptr"].amin() >= 0


@pytest.mark.parametrize(("mode", "epoch"), (("off", 1), ("parent_only", 6), ("full", 11)))
def test_real_decoder_aux_satisfies_strict_mode_loss_contract(
    decoder_inputs, mode, epoch
):
    model, features, memory = decoder_inputs
    outputs, aux = model(
        features,
        memory=None if mode == "off" else memory,
        pc_mode=mode,
        epoch=epoch,
        return_aux=True,
        query_image_ids=["query"],
    )
    gt = (torch.rand(1, 1, 98, 98) > 0.5).float()
    loss, metrics = pc_hbm_labeled_loss(
        outputs,
        aux,
        gt,
        epoch,
        model.pc_cfg,
        pc_mode=mode,
        strict=True,
    )
    assert torch.isfinite(loss)
    assert "L_base" in metrics


def test_real_teacher_aux_builds_soft_targets_and_corrected_features(
    decoder_inputs,
):
    model, features, memory = decoder_inputs
    _, aux = model(
        features,
        memory=memory,
        pc_mode="teacher_pseudo",
        epoch=31,
        return_aux=True,
        query_image_ids=["unlabeled-query"],
    )
    pseudo = prepare_pseudo_targets(aux, model.pc_cfg, strict=True)
    assert pseudo["p_soft"].shape == (1, 1, 98, 98)
    assert pseudo["confidence"].shape == (1, 1, 98, 98)
    assert pseudo["confidence"].amin() >= 0
    assert pseudo["confidence"].amax() <= 1
    assert pseudo["hard_target"].shape == (1, 1, 98, 98)
    assert pseudo["hard_valid"].shape == (1, 1, 98, 98)
    torch.testing.assert_close(
        pseudo["hard_weight"],
        pseudo["confidence"] * pseudo["hard_valid"],
    )
    distill = pseudo["distill_features"]
    assert distill["p3_corr"].shape == (1, 128, 28, 28)
    assert distill["p2_refined"].shape == (1, 128, 28, 28)
