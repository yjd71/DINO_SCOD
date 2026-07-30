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


def test_lite_config_rejects_changes_to_fixed_contract() -> None:
    with pytest.raises(ValueError, match="route_global_weight"):
        DinoPCHBMConfig(route_global_weight=0.6)
    with pytest.raises(ValueError, match="region_max_quota"):
        DinoPCHBMConfig(region_max_quota=(32, 32))
    with pytest.raises(ValueError, match="child_window_size"):
        DinoPCHBMConfig(child_window_size=5)
    with pytest.raises(ValueError, match="temperatures"):
        DinoPCHBMConfig(tau_child=float("nan"))
    with pytest.raises(ValueError, match="loss weights"):
        DinoPCHBMConfig(lambda_pair=float("inf"))


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
