from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.PC_HBM.dino_memory_builder import DinoMemoryBuilder
from Model.PC_HBM.memory import PCMemory
from Model.PC_HBM.routing import CamouflageContextRouter


def _entry(image_ids: tuple[str, ...], *, dim: int = 128) -> dict:
    count = len(image_ids)
    pair_meta = [
        {"image_id": image_id, "region_id": region_id}
        for image_id in image_ids
        for region_id in (0, 1)
    ]
    pair_count = len(pair_meta)
    return {
        "source": "labeled_only",
        "route": {
            "global_keys": F.normalize(torch.randn(count, dim), dim=-1),
            "environment_keys": F.normalize(torch.randn(count, dim), dim=-1),
            "img_ids": list(image_ids),
        },
        "pairs": {
            "p3_keys": F.normalize(torch.randn(pair_count, dim), dim=-1),
            "p2_keys": F.normalize(torch.randn(pair_count, dim), dim=-1),
            "region_ids": torch.tensor([0, 1] * count),
            "pair_meta": pair_meta,
        },
    }


def _ready_memory() -> tuple[PCMemory, DinoPCHBMConfig]:
    cfg = DinoPCHBMConfig()
    memory = PCMemory(config=cfg)
    memory.append((_entry(("A", "B")), _entry(("C",))))
    memory.finalize(compat_meta=cfg.expected_memory_meta())
    assert memory.is_ready()
    return memory, cfg


def test_lite_config_contract_and_stage_schedules() -> None:
    cfg = DinoPCHBMConfig()
    assert cfg.expected_memory_meta() == {
        "architecture": "DINO_SCOD_PC_HBM_LITE",
        "schema_version": 2,
        "input_size": 392,
        "token_hw": (28, 28),
        "output_hw": (98, 98),
        "dino_layer_indices": (2, 5, 8, 11),
        "encoder_dim": 768,
        "decoder_dim": 128,
        "memory_dim": 128,
        "child_window_size": 3,
        "region_names": ("fg_boundary", "bg_near"),
        "storage_dtype": "float16",
        "source": "labeled_only",
        "route_environment_min_mass": 1.0e-3,
        "fg_boundary_kernel": 3,
        "bg_near_kernel": 7,
        "gt_binary_threshold": 0.5,
        "region_max_quota": (784, 784),
        "region_min_quota": (0, 0),
        "region_sampling_ratio": (1.0, 1.0),
    }
    assert [cfg.pc_mode_for_epoch(epoch) for epoch in (1, 5, 6, 10, 11)] == [
        "off",
        "off",
        "verify_only",
        "verify_only",
        "full",
    ]
    assert [cfg.injection_scale(epoch) for epoch in (10, 11, 12, 13)] == [
        0.0,
        pytest.approx(1.0 / 3.0),
        pytest.approx(2.0 / 3.0),
        1.0,
    ]
    cfg.configure_training_design("teacher_only")
    assert [cfg.pc_mode_for_epoch(epoch) for epoch in (1, 5, 6)] == [
        "verify_only",
        "verify_only",
        "full",
    ]
    with pytest.raises(ValueError, match="Unsupported"):
        cfg.configure_training_design("unsupported")


def test_lite_config_accepts_tunable_hyperparameters() -> None:
    cfg = DinoPCHBMConfig(
        enabled=False,
        input_size=420,
        encoder_dim=768,
        decoder_dim=256,
        token_size=30,
        output_size=105,
        dino_layer_indices=(10, 1, 7, 4),
        memory_dim=256,
        memory_source="labeled_only",
        use_unlabeled_memory_update=False,
        memory_storage_dtype="bf16",
        memory_device="cpu",
        memory_format_version=2,
        memory_schema_version=2,
        memory_architecture="DINO_SCOD_PC_HBM_LITE",
        exclude_self_match=False,
        route_top_img_k=6,
        route_global_weight=0.7,
        route_environment_weight=0.3,
        route_environment_min_mass=0.02,
        p3_top_ratio=0.2,
        p3_min_tokens=12,
        p3_max_tokens=120,
        query_boundary_weight=0.8,
        query_uncertainty_weight=0.2,
        fg_boundary_kernel=5,
        bg_near_kernel=9,
        gt_binary_threshold=0.4,
        region_names=("edge", "near"),
        region_max_quota=(60, 72),
        region_min_quota=(4, 6),
        region_sampling_ratio=(0.25, 0.75),
        parent_topk_per_region=6,
        query_chunk_size=128,
        child_window_size=5,
        tau_parent=0.12,
        tau_child=0.18,
        child_mix_init_logit=0.4,
        child_verification_mode="parent_conditioned",
        verification_strength_init=0.3,
        verification_logit_clip=5.0,
        relation_norm_eps=2.0e-4,
        lambda_candidate_verify=0.6,
        verify_start_epoch=2,
        full_pc_start_epoch=4,
        teacher_only_full_start_epoch=3,
        pc_injection_ramp_epochs=2,
        lambda_pair=0.35,
        lambda_u=0.8,
        feature_distill_p3_weight=0.1,
        use_amp=False,
        grad_clip_norm=2.5,
        ema_momentum=0.98,
        diagnostic_window_epochs=5,
        warn_low_pair_valid_ratio=0.1,
        warn_pair_acc_near_random=0.15,
        warn_gate_inactive_threshold=0.08,
        warn_delta_large_threshold=1.5,
    )

    assert cfg.p3_top_ratio == pytest.approx(0.2)
    assert cfg.route_global_weight == pytest.approx(0.7)
    assert cfg.region_max_quota == (60, 72)
    assert cfg.child_window_size == 5
    assert cfg.child_verification_mode == "parent_conditioned"
    assert cfg.verification_strength_init == pytest.approx(0.3)
    assert cfg.lambda_candidate_verify == pytest.approx(0.6)
    assert cfg.memory_storage_dtype == "bfloat16"
    assert cfg.dino_layer_indices == (1, 4, 7, 10)
    assert cfg.expected_memory_meta() == {
        "architecture": "DINO_SCOD_PC_HBM_LITE",
        "schema_version": 2,
        "input_size": 420,
        "token_hw": (30, 30),
        "output_hw": (105, 105),
        "dino_layer_indices": (1, 4, 7, 10),
        "encoder_dim": 768,
        "decoder_dim": 256,
        "memory_dim": 256,
        "child_window_size": 5,
        "region_names": ("edge", "near"),
        "storage_dtype": "bfloat16",
        "source": "labeled_only",
        "route_environment_min_mass": 0.02,
        "fg_boundary_kernel": 5,
        "bg_near_kernel": 9,
        "gt_binary_threshold": 0.4,
        "region_max_quota": (60, 72),
        "region_min_quota": (4, 6),
        "region_sampling_ratio": (0.25, 0.75),
    }
    cfg.configure_training_design("two_stage")
    assert [cfg.pc_mode_for_epoch(epoch) for epoch in (1, 2, 3, 4)] == [
        "off",
        "verify_only",
        "verify_only",
        "full",
    ]
    assert [cfg.injection_scale(epoch) for epoch in (3, 4, 5)] == [
        0.0,
        pytest.approx(0.5),
        1.0,
    ]
    cfg.configure_training_design("teacher_only")
    assert [cfg.pc_mode_for_epoch(epoch) for epoch in (1, 2, 3)] == [
        "verify_only",
        "verify_only",
        "full",
    ]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"child_verification_mode": "unknown"}, "child_verification_mode"),
        ({"verification_strength_init": 1.0}, "verification_strength_init"),
        ({"verification_logit_clip": 0.0}, "verification_logit_clip"),
        ({"relation_norm_eps": 0.0}, "relation_norm_eps"),
        ({"lambda_candidate_verify": -0.1}, "lambda_candidate_verify"),
    ],
)
def test_child_verifier_v3_config_rejects_invalid_values(
    kwargs: dict, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        DinoPCHBMConfig(**kwargs)


def test_lite_config_accepts_safe_zero_and_closed_boundary_values() -> None:
    cfg = DinoPCHBMConfig(
        decoder_dim=130,
        memory_dim=130,
        p3_top_ratio=0.0,
        p3_min_tokens=0,
        p3_max_tokens=0,
        route_environment_min_mass=0.0,
        grad_clip_norm=0.0,
        ema_momentum=1.0,
        warn_delta_large_threshold=0.0,
    )
    assert cfg.decoder_dim == cfg.memory_dim == 130
    assert cfg.p3_top_ratio == 0.0
    assert cfg.p3_min_tokens == cfg.p3_max_tokens == 0
    assert cfg.route_environment_min_mass == 0.0
    assert cfg.grad_clip_norm == 0.0
    assert cfg.ema_momentum == 1.0
    assert cfg.warn_delta_large_threshold == 0.0


@pytest.mark.parametrize(
    ("configured", "canonical", "dtype"),
    [
        ("fp16", "float16", torch.float16),
        ("bf16", "bfloat16", torch.bfloat16),
        ("fp32", "float32", torch.float32),
    ],
)
def test_memory_storage_dtype_is_tunable_on_cpu(
    configured: str,
    canonical: str,
    dtype: torch.dtype,
) -> None:
    cfg = DinoPCHBMConfig(memory_storage_dtype=configured)
    memory = PCMemory(config=cfg)
    memory.append(_entry(("A",)))
    memory.finalize(compat_meta=cfg.expected_memory_meta())
    assert cfg.memory_storage_dtype == canonical
    assert memory.route["global_keys"].device.type == "cpu"
    assert memory.route["global_keys"].dtype == dtype
    assert memory.pairs["p3_keys"].dtype == dtype
    restored = PCMemory(config=cfg)
    restored.load_state_dict(memory.state_dict())
    assert restored.route["environment_keys"].dtype == dtype
    assert restored.pairs["p2_keys"].dtype == dtype


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"input_size": 393}, "input_size"),
        ({"token_size": 29}, "input_size"),
        ({"output_size": 99}, "output_size"),
        ({"encoder_dim": 1024}, "encoder_dim"),
        ({"decoder_dim": 130}, "memory_dim"),
        ({"dino_layer_indices": (1, 2, 3)}, "exactly 4"),
        ({"dino_layer_indices": (1, 3, 3, 8)}, "unique"),
        ({"dino_layer_indices": (1, 4, 8, 12)}, r"\[0,11\]"),
        ({"route_top_img_k": True}, "route_top_img_k"),
        (
            {
                "route_global_weight": 0.0,
                "route_environment_weight": 0.0,
            },
            "route weights",
        ),
        ({"p3_top_ratio": float("nan")}, "p3_top_ratio"),
        ({"p3_top_ratio": 1.01}, "p3_top_ratio"),
        ({"p3_min_tokens": 65, "p3_max_tokens": 64}, "p3_min_tokens"),
        ({"p3_max_tokens": 785}, "p3_max_tokens"),
        ({"query_boundary_weight": float("inf")}, "query_boundary_weight"),
        ({"fg_boundary_kernel": 4}, "fg_boundary_kernel"),
        ({"gt_binary_threshold": -0.1}, "gt_binary_threshold"),
        ({"region_names": ("same", "same")}, "unique"),
        ({"region_names": ("pair_union", "near")}, "reserved"),
        ({"region_max_quota": (48,)}, "exactly 2"),
        (
            {
                "region_min_quota": (16, 8),
                "region_max_quota": (8, 48),
            },
            r"region_min_quota\[0\]",
        ),
        ({"region_sampling_ratio": (0.5, float("inf"))}, "finite"),
        ({"child_window_size": 2}, "child_window_size"),
        ({"tau_child": 0.0}, "tau_child"),
        ({"child_mix_init_logit": float("nan")}, "child_mix_init_logit"),
        (
            {"verify_start_epoch": 5, "full_pc_start_epoch": 4},
            "full_pc_start_epoch",
        ),
        ({"pc_injection_ramp_epochs": 0}, "pc_injection_ramp_epochs"),
        ({"lambda_pair": float("inf")}, "lambda_pair"),
        ({"warn_low_pair_valid_ratio": float("nan")}, "finite"),
        ({"memory_storage_dtype": "int8"}, "memory_storage_dtype"),
        ({"memory_source": "pseudo"}, "labeled_only"),
        ({"use_unlabeled_memory_update": True}, "must be False"),
        ({"memory_device": "cuda"}, "requires memory_device"),
        ({"memory_format_version": 3}, "requires memory_format_version"),
        ({"memory_schema_version": 3}, "requires memory_schema_version"),
        ({"memory_architecture": "CUSTOM"}, "requires memory_architecture"),
    ],
)
def test_lite_config_rejects_invalid_types_ranges_and_relations(
    kwargs: dict,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        DinoPCHBMConfig(**kwargs)


def test_append_multiple_batches_finalizes_as_cpu_fp16() -> None:
    memory, _ = _ready_memory()
    assert memory.route["img_ids"] == ["A", "B", "C"]
    for group in (memory.route, memory.pairs):
        for value in group.values():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                assert value.device.type == "cpu"
                assert value.dtype == torch.float16
                assert value.is_contiguous()
    assert memory.pairs["p3_keys"].shape == memory.pairs["p2_keys"].shape
    assert set(memory.pairs["region_ids"].tolist()) == {0, 1}
    assert memory.pair_img_to_indices["B"].tolist() == [2, 3]


def test_duplicate_route_id_and_unlabeled_entry_are_rejected() -> None:
    memory = PCMemory()
    memory.append(_entry(("A",)))
    with pytest.raises(ValueError, match="Duplicate image IDs"):
        memory.append(_entry(("A",)))
    rejected = _entry(("B",))
    rejected["source"] = "pseudo"
    with pytest.raises(ValueError, match="labeled"):
        memory.append(rejected)


@pytest.mark.parametrize(
    ("group", "extra_key"),
    (("route", "route_embed"), ("pairs", "child_ptr")),
)
def test_append_rejects_unknown_v1_fields(
    group: str,
    extra_key: str,
) -> None:
    memory = PCMemory()
    entry = _entry(("A",))
    entry[group][extra_key] = torch.empty(0)
    with pytest.raises(ValueError, match="contain exactly"):
        memory.append(entry)


def test_one_sided_pair_memory_is_not_ready() -> None:
    cfg = DinoPCHBMConfig()
    memory = PCMemory(config=cfg)
    entry = _entry(("A",))
    entry["pairs"]["p3_keys"] = entry["pairs"]["p3_keys"][:1]
    entry["pairs"]["p2_keys"] = entry["pairs"]["p2_keys"][:1]
    entry["pairs"]["region_ids"] = torch.tensor([0])
    entry["pairs"]["pair_meta"] = [{"image_id": "A", "region_id": 0}]
    memory.append(entry)
    memory.finalize(compat_meta=cfg.expected_memory_meta())
    assert not memory.is_ready()


def test_state_round_trip_and_v1_rejection_are_atomic() -> None:
    memory, cfg = _ready_memory()
    state = memory.state_dict()
    assert set(state) == {
        "format_version",
        "schema_version",
        "compat_meta",
        "memory_dim",
        "storage_dtype",
        "route",
        "pairs",
        "finalized",
    }
    restored = PCMemory(config=cfg)
    restored.load_state_dict(state)
    assert restored.is_ready()
    torch.testing.assert_close(restored.route["global_keys"], memory.route["global_keys"])
    torch.testing.assert_close(restored.pairs["p2_keys"], memory.pairs["p2_keys"])
    assert restored.pairs["pair_meta"] == memory.pairs["pair_meta"]

    before = restored.state_dict()
    incompatible = copy.deepcopy(state)
    incompatible["format_version"] = 1
    incompatible["schema_version"] = 1
    with pytest.raises(ValueError, match="compat_mismatch"):
        restored.load_state_dict(incompatible)
    assert restored.is_ready()
    torch.testing.assert_close(
        restored.route["global_keys"],
        before["route"]["global_keys"],
    )
    assert restored.route["img_ids"] == before["route"]["img_ids"]


def test_route_and_pair_image_consistency_is_enforced() -> None:
    cfg = DinoPCHBMConfig()
    memory = PCMemory(config=cfg)
    entry = _entry(("A",))
    entry["pairs"]["pair_meta"][0]["image_id"] = "missing"
    memory.append(entry)
    with pytest.raises(ValueError, match="route table"):
        memory.finalize(compat_meta=cfg.expected_memory_meta())
    assert not memory.is_ready()


def test_self_match_exclusion_and_pair_subbank_indices() -> None:
    memory, _ = _ready_memory()
    query = memory.route["global_keys"][0:1].float()
    environment = memory.route["environment_keys"][0:1].float()
    routed = memory.route_query(
        query,
        environment,
        top_img_k=4,
        query_image_ids=["A"],
        exclude_self_match=True,
    )
    assert "A" not in routed["top_img_ids"][0]
    assert routed["top_img_valid"].sum().item() == 2

    subbank = memory.get_pair_subbank(["B", "C"], dtype=torch.float32)
    assert subbank["pair_indices"].tolist() == [2, 3, 4, 5]
    assert subbank["region_ids"].tolist() == [0, 1, 0, 1]
    assert {item["image_id"] for item in subbank["pair_meta"]} == {"B", "C"}
    assert subbank["p3_keys"].dtype == torch.float32


class _IdentityParent:
    @staticmethod
    def encode_k_map(p3: torch.Tensor) -> torch.Tensor:
        return p3


class _TokenChild:
    @staticmethod
    def encode_child_map(
        p2: torch.Tensor,
        batch_ids: torch.Tensor,
        flat_indices: torch.Tensor,
        p3_hw: tuple[int, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        assert p3_hw == p2.shape[-2:]
        tokens = p2.flatten(2).transpose(1, 2)
        return {
            "q_child": tokens[batch_ids, flat_indices],
            "child_patches": p2.new_empty((flat_indices.numel(), p2.size(1), 3, 3)),
        }


def test_memory_builder_uses_aligned_p3_p2_pairs_and_no_m2() -> None:
    cfg = DinoPCHBMConfig()
    builder = DinoMemoryBuilder(
        cfg,
        CamouflageContextRouter(),
        _IdentityParent(),
        _TokenChild(),
    )
    p3 = torch.randn(2, 128, 28, 28)
    p2 = torch.randn(2, 128, 28, 28)
    gt = torch.zeros(2, 1, 28, 28)
    gt[0, 0, 7:20, 8:19] = 1
    gt[1, 0, 10:17, 11:18] = 1
    features = {
        "x3": torch.randn_like(p3),
        "p3": p3,
        "p2": p2,
        "m3": torch.randn(2, 1, 28, 28),
    }
    with pytest.raises(KeyError, match="unsupported keys"):
        builder(
            {**features, "m2": torch.randn(2, 1, 28, 28)},
            gt,
            ["A", "B"],
        )
    entries = builder(
        features,
        gt,
        ["A", "B"],
    )
    assert set(entries) == {"source", "route", "pairs"}
    assert set(entries["route"]) == {"global_keys", "environment_keys", "img_ids"}
    assert set(entries["pairs"]) == {"p3_keys", "p2_keys", "region_ids", "pair_meta"}
    assert entries["pairs"]["p3_keys"].shape == entries["pairs"]["p2_keys"].shape
    assert entries["pairs"]["p3_keys"].size(0) == len(entries["pairs"]["pair_meta"])
    for index, metadata in enumerate(entries["pairs"]["pair_meta"]):
        batch_index = 0 if metadata["image_id"] == "A" else 1
        flat_index = metadata["flat_index"]
        torch.testing.assert_close(
            entries["pairs"]["p3_keys"][index],
            p3.flatten(2).transpose(1, 2)[batch_index, flat_index],
        )
        torch.testing.assert_close(
            entries["pairs"]["p2_keys"][index],
            p2.flatten(2).transpose(1, 2)[batch_index, flat_index],
        )
