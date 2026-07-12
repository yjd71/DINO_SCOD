from __future__ import annotations

import copy

import pytest
import torch

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.PC_HBM.memory import PCMemory, build_pc_regions, sample_region_indices


def _entry(image_ids: tuple[str, ...] = ("A", "B"), dim: int = 128) -> dict:
    image_count = len(image_ids)
    parent_count = image_count * 2
    route_embed = torch.randn(image_count, dim)
    route = {
        name: torch.randn(image_count, dim)
        for name in (
            "x3_global",
            "x3_boundary",
            "x3_uncertain",
            "x3_bg_near",
            "x3_environment",
        )
    }
    route["route_embed"] = route_embed
    route["img_ids"] = list(image_ids)
    parent_meta = []
    child_meta = []
    for image_id in image_ids:
        for region in ("fg_boundary", "bg_near"):
            parent_meta.append({"image_id": image_id, "region": region})
            child_meta.append({"image_id": image_id, "region": region})
    return {
        "source": "labeled_only",
        "route": route,
        "parent": {
            "p3_keys": torch.randn(parent_count, dim),
            "p3_values": torch.randn(parent_count, 8),
            "p3_geometry": torch.randn(parent_count, 6),
            "child_ptr": torch.arange(parent_count),
            "parent_meta": parent_meta,
        },
        "child": {
            "p2_child_keys": torch.randn(parent_count, dim),
            "p2_child_geo": torch.randn(parent_count, 6),
            "child_meta": child_meta,
        },
    }


def _ready_memory() -> tuple[PCMemory, DinoPCHBMConfig]:
    config = DinoPCHBMConfig()
    memory = PCMemory(config.memory_dim, config.value_dim, config.geometry_dim)
    memory.append(_entry())
    memory.finalize(compat_meta=config.expected_memory_meta(producer_fingerprint="ema-1"))
    return memory, config


def test_memory_is_labeled_only_cpu_fp16_and_dtype_aware() -> None:
    memory, _ = _ready_memory()
    assert memory.is_ready()
    for group in (memory.route, memory.parent, memory.child):
        for value in group.values():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                assert value.device.type == "cpu"
                assert value.dtype == torch.float16

    subbank = memory.get_parent_subbank(["A"], dtype=torch.float64)
    assert subbank["p3_keys"].dtype == torch.float64
    assert subbank["p3_values"].dtype == torch.float64
    assert subbank["p3_geometry"].dtype == torch.float64
    assert {item["image_id"] for item in subbank["parent_meta"]} == {"A"}

    pointers = torch.tensor([[0, -1, 999]], dtype=torch.long)
    children = memory.get_child_by_ptr(pointers, dtype=torch.float32)
    assert children["p2_child_keys"].dtype == torch.float32
    assert children["child_valid"].tolist() == [[True, False, False]]
    torch.testing.assert_close(children["p2_child_keys"][0, 1:], torch.zeros(2, 128))


def test_memory_rejects_unlabeled_entries() -> None:
    memory = PCMemory()
    entry = _entry(("unlabeled",))
    entry["source"] = "pseudo_unlabeled"
    with pytest.raises(ValueError, match="labeled_only"):
        memory.append(entry)


def test_memory_round_trip_and_compatibility_contract() -> None:
    memory, config = _ready_memory()
    state = copy.deepcopy(memory.state_dict())
    restored = PCMemory()
    restored.load_state_dict({"memory": state, "compat_meta": state["compat_meta"]})
    assert restored.is_ready()
    ok, reason = restored.validate_compat(config.expected_memory_meta())
    assert ok and reason is None
    mismatch = config.expected_memory_meta()
    mismatch["token_hw"] = (14, 14)
    result = restored.validate_compat(mismatch)
    assert not result
    assert result.reason == "compat_mismatch:token_hw"
    assert restored.state_dict()["route"]["route_embed"].dtype == torch.float16


def test_regions_are_nearest_binary_mutually_exclusive_and_use_signed_distance() -> None:
    gt = torch.zeros(1, 1, 56, 56)
    gt[:, :, 14:42, 14:42] = 1.0
    regions = build_pc_regions(gt, (28, 28))
    partition = torch.cat(
        [regions[name] for name in ("fg_core", "fg_boundary", "bg_near", "bg_far")],
        dim=1,
    )
    torch.testing.assert_close(partition.sum(dim=1), torch.ones(1, 28, 28))
    assert torch.all((regions["fg"] == 0) | (regions["fg"] == 1))
    assert regions["sdf"][0, 0, 14, 14] > 0
    assert regions["sdf"][0, 0, 0, 0] < 0
    geometry = regions["geometry"]
    assert geometry.shape == (1, 6, 28, 28)
    assert torch.isfinite(geometry).all()
    expected_reliability = torch.exp(-regions["sdf"].abs() / 0.15)
    torch.testing.assert_close(geometry[:, 5:6], expected_reliability)


def test_region_sampling_is_deterministic_bounded_and_never_duplicates() -> None:
    mask = torch.ones(28, 28, dtype=torch.bool)
    score = torch.arange(28 * 28, dtype=torch.float32).reshape(28, 28)
    first = sample_region_indices(mask, score, "fg_boundary")
    second = sample_region_indices(mask, score, "fg_boundary")
    assert torch.equal(first, second)
    assert first.numel() == 64
    assert first.unique().numel() == first.numel()
    empty = sample_region_indices(torch.zeros_like(mask), score, "fg_boundary")
    assert empty.numel() == 0

