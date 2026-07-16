from __future__ import annotations

import pytest
import torch

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.bgfbr_decoder import BGFBRDecoder
from Model.decoder import build_decoder
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
def bgfbr_inputs():
    torch.manual_seed(23)
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
    decoder = build_decoder(pc_cfg=config).eval()
    features = [torch.randn(1, 28 * 28, 768) for _ in range(4)]
    image_rgb = torch.rand(1, 3, 392, 392)
    return config, decoder, features, image_rgb, _ready_memory(config)


def test_factory_selects_bgfbr_and_raw_student_keeps_the_same_base_contract():
    config = DinoPCHBMConfig()
    teacher = build_decoder(pc_cfg=config, attach_pc=True)
    raw_student = build_decoder(pc_cfg=config, attach_pc=False)

    assert isinstance(teacher, BGFBRDecoder)
    assert teacher.pc_hbm is not None
    assert isinstance(raw_student, BGFBRDecoder)
    assert raw_student.pc_hbm is None
    shared_teacher_keys = {
        key for key in teacher.state_dict() if not key.startswith("pc_hbm.")
    }
    assert shared_teacher_keys == set(raw_student.state_dict())


def test_off_mode_requires_rgb_and_keeps_five_output_aux_contract(bgfbr_inputs):
    _, decoder, features, image_rgb, _ = bgfbr_inputs
    with pytest.raises(ValueError, match="requires image_rgb"):
        decoder(features, None, pc_mode="off")

    outputs, aux = decoder(
        features, image_rgb, pc_mode="off", return_aux=True
    )
    assert all(tuple(value.shape) == (1, 1, 98, 98) for value in outputs)
    assert aux["decoder_architecture"] == "bgfbr_pc_v1"
    assert aux["z_final"] is aux["z_main"]
    assert aux["pc_active"] is False
    assert tuple(aux["bgfbr"]["cam_feat"].shape) == (1, 128, 28, 28)
    assert tuple(aux["bgfbr"]["edge_full"].shape) == (1, 1, 392, 392)
    assert len(aux["bgfbr"]["fg_output"]) == 4
    assert len(aux["bgfbr"]["bg_output"]) == 4


@pytest.mark.parametrize(
    ("mode", "mixture_skipped", "z_final_is_none"),
    (
        ("parent_only", True, False),
        ("full", False, False),
        ("teacher_pseudo", False, False),
        ("student_core", True, True),
    ),
)
def test_all_pc_modes_preserve_bgfbr_and_ts_contracts(
    bgfbr_inputs, mode, mixture_skipped, z_final_is_none
):
    _, decoder, features, image_rgb, memory = bgfbr_inputs
    outputs, aux = decoder(
        features,
        image_rgb,
        memory=memory,
        pc_mode=mode,
        epoch=11,
        return_aux=True,
    )
    assert all(tuple(value.shape) == (1, 1, 98, 98) for value in outputs)
    assert aux["pc_active"] is True
    assert aux["mixture_skipped"] is mixture_skipped
    assert (aux["z_final"] is None) is z_final_is_none
    if mode == "teacher_pseudo":
        assert set(aux["distill_features"]["p1"]) == {
            "B1",
            "G1_raw_map",
            "R1_map",
            "O1_map",
            "R_sup_map",
            "valid1_map",
        }
    if mode == "student_core":
        assert aux["p1_pra"] is not None
        assert aux["mixture"] is None


def test_bgfbr_pc_channels_bridge_and_memory_features(bgfbr_inputs):
    _, decoder, features, image_rgb, _ = bgfbr_inputs
    assert decoder.pc_hbm.boundary3.net[0].in_channels == 7
    assert decoder.pc_hbm.p2_bra.boundary_head.net[0].in_channels == 10
    assert decoder.pc_hbm.p1_pra.boundary_head.net[0].in_channels == 10
    assert decoder.pc_hbm.mixture.mix_head[0].in_channels == 16
    assert decoder.pc_hbm.p3_p2_bridge.gate.in_channels == 257

    memory_features = decoder.forward_memory_features(features, image_rgb)
    assert set(memory_features) == {"x3", "p3", "p2", "m3", "m2"}
    assert tuple(memory_features["x3"].shape) == (1, 128, 28, 28)
    assert tuple(memory_features["p2"].shape) == (1, 128, 28, 28)
    assert tuple(memory_features["m2"].shape) == (1, 1, 28, 28)


def test_missing_memory_is_an_explicit_inference_style_fallback(bgfbr_inputs):
    _, decoder, features, image_rgb, _ = bgfbr_inputs
    decoder.eval()
    _, aux = decoder(
        features,
        image_rgb,
        memory=None,
        pc_mode="full",
        epoch=11,
        return_aux=True,
    )
    assert aux["pc_active"] is False
    assert aux["fallback_reason"] == "memory_missing"
    assert torch.equal(aux["z_final"], aux["z_main"])


def test_missing_memory_fails_fast_while_training(bgfbr_inputs):
    _, decoder, features, image_rgb, _ = bgfbr_inputs
    decoder.train()
    with pytest.raises(RuntimeError, match="training requires"):
        decoder(
            features,
            image_rgb,
            memory=None,
            pc_mode="full",
            epoch=11,
        )
