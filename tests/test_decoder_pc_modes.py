from __future__ import annotations

import pytest
import torch

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.decoder import Decoder
from Model.PC_HBM.memory import PCMemory


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


def test_missing_or_incompatible_memory_returns_explicit_baseline_fallback(
    decoder_inputs,
):
    model, features, memory = decoder_inputs
    baseline = model(features)
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
