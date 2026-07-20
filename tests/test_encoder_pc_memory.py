from __future__ import annotations

import copy
from collections.abc import Mapping

import pytest
import torch

from Model.PC_HBM.encoder.encoder_memory import (
    ENCODER_PC_MEMORY_ARCHITECTURE,
    ENCODER_PC_MEMORY_SCHEMA_VERSION,
    EncoderPCMemory,
    build_encoder_memory_compat_meta,
)


def _compat(
    producer: str = "adapter-sha256",
    split: str = "labeled-split-sha256",
) -> dict:
    return build_encoder_memory_compat_meta(
        producer_fingerprint=producer,
        labeled_split_fingerprint=split,
    )


def _entry(image_ids: tuple[str, ...] = ("A", "B")) -> dict:
    image_count = len(image_ids)
    parent_count = image_count * 2
    child_count = parent_count
    reliability = torch.linspace(0.5, 0.8, parent_count)
    values = torch.zeros(parent_count, 8)
    values[torch.arange(parent_count), torch.arange(parent_count) % 4] = 1.0
    values[:, 4] = (torch.arange(parent_count) % 2 == 0).float()
    values[:, 5] = 1.0 - values[:, 4]
    values[:, 6] = torch.linspace(-1.0, 1.0, parent_count)
    values[:, 7] = reliability
    image_index = torch.arange(parent_count, dtype=torch.long) // 2
    return {
        "source": "labeled_only",
        "route": {
            "route_keys": torch.randn(image_count, 128),
            "cls4_keys": torch.randn(image_count, 128),
            "f4_global_keys": torch.randn(image_count, 128),
            "f3_boundary_keys": torch.randn(image_count, 128),
            "image_ids": list(image_ids),
        },
        "parent": {
            "f3_parent_keys": torch.randn(parent_count, 128),
            "values": values,
            "geometry": torch.randn(parent_count, 6),
            "child_ptr": torch.arange(child_count, dtype=torch.long),
            "image_index": image_index,
            "region_id": torch.arange(parent_count, dtype=torch.long) % 4,
            "flat_index": torch.arange(parent_count, dtype=torch.long) + 10,
            "reliability": reliability,
        },
        "child": {
            "f2_child_keys": torch.randn(child_count, 128),
            "f1_detail_keys": torch.randn(child_count, 128),
            "geometry": torch.randn(child_count, 6),
            "image_index": image_index.clone(),
            "flat_index": torch.arange(child_count, dtype=torch.long) + 20,
        },
    }


def _ready_memory() -> EncoderPCMemory:
    memory = EncoderPCMemory()
    memory.append(_entry())
    memory.finalize(compat_meta=_compat())
    return memory


def test_schema_v3_is_labeled_only_cpu_fp16_and_tensorized() -> None:
    memory = _ready_memory()

    assert memory.is_ready()
    assert memory.num_images == 2
    assert memory.num_parents == 4
    assert memory.num_children == 4
    assert memory.compat_meta["architecture"] == ENCODER_PC_MEMORY_ARCHITECTURE
    assert memory.compat_meta["schema_version"] == ENCODER_PC_MEMORY_SCHEMA_VERSION
    assert memory.compat_meta["source"] == "labeled_only"
    assert memory.route["image_ids"] == ["A", "B"]

    for group in (memory.route, memory.parent, memory.child):
        for name, value in group.items():
            if name == "image_ids":
                assert all(isinstance(item, str) for item in value)
                continue
            assert isinstance(value, torch.Tensor)
            assert value.device.type == "cpu"
            assert not value.requires_grad
            if value.is_floating_point():
                assert value.dtype == torch.float16

    assert memory.parent["child_ptr"].dtype == torch.int32
    assert memory.parent["image_index"].dtype == torch.int32
    assert memory.parent["region_id"].dtype == torch.int16
    assert memory.parent["flat_index"].dtype == torch.int16
    assert memory.parent["reliability"].dtype == torch.float16
    assert memory.child["image_index"].dtype == torch.int32
    assert memory.child["flat_index"].dtype == torch.int16
    assert "parent_meta" not in memory.parent
    assert "child_meta" not in memory.child
    assert all(
        not isinstance(item, Mapping)
        for group in (memory.route, memory.parent, memory.child)
        for value in group.values()
        if isinstance(value, list)
        for item in value
    )


def test_append_offsets_local_image_indices_and_child_pointers() -> None:
    memory = EncoderPCMemory()
    memory.append(_entry(("A", "B")))
    memory.append(_entry(("C", "D")))
    memory.finalize(compat_meta=_compat())

    assert memory.route["image_ids"] == ["A", "B", "C", "D"]
    assert memory.parent["image_index"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert memory.child["image_index"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert memory.parent["child_ptr"].tolist() == list(range(8))


def test_unlabeled_entries_and_non_fp16_storage_are_rejected() -> None:
    entry = _entry()
    entry["source"] = "unlabeled"
    with pytest.raises(ValueError, match="labeled entries only"):
        EncoderPCMemory().append(entry)

    with pytest.raises(ValueError, match="fixed to float16"):
        EncoderPCMemory(storage_dtype=torch.float32)
    with pytest.raises(ValueError, match="CPU float16"):
        memory = EncoderPCMemory()
        memory.append(_entry())
        memory.finalize(dtype=torch.float32)


def test_state_dict_round_trip_is_strict_and_does_not_alias_storage() -> None:
    memory = _ready_memory()
    state = memory.state_dict()
    restored = EncoderPCMemory.from_state_dict(state)

    assert restored.is_ready()
    assert restored.compat_meta == memory.compat_meta
    for group_name in ("route", "parent", "child"):
        original_group = getattr(memory, group_name)
        restored_group = getattr(restored, group_name)
        assert set(restored_group) == set(original_group)
        for name, original in original_group.items():
            if isinstance(original, torch.Tensor):
                assert torch.equal(restored_group[name], original)
                assert restored_group[name].data_ptr() != original.data_ptr()
            else:
                assert restored_group[name] == original
                assert restored_group[name] is not original

    state["parent"]["f3_parent_keys"].zero_()
    assert torch.count_nonzero(memory.parent["f3_parent_keys"]) > 0


@pytest.mark.parametrize("schema_version", [1, 2])
def test_decoder_side_schema_v1_v2_are_explicitly_rejected(schema_version: int) -> None:
    old_state = {
        "format_version": schema_version,
        "schema_version": schema_version,
        "compat_meta": {"schema_version": schema_version},
    }
    with pytest.raises(
        RuntimeError,
        match="Decoder-side PC-HBM memory is incompatible with encoder-side schema v3",
    ):
        EncoderPCMemory().load_state_dict(old_state)


def test_loader_rejects_conflicting_schema_and_legacy_prototype_dicts() -> None:
    state = _ready_memory().state_dict()
    conflicting = {"schema_version": 2, "memory": state}
    with pytest.raises(RuntimeError, match="Conflicting encoder PC-HBM memory schema"):
        EncoderPCMemory().load_state_dict(conflicting)

    with_legacy_meta = copy.deepcopy(state)
    with_legacy_meta["parent"]["parent_meta"] = [{"image_id": "A"}]
    with pytest.raises(ValueError, match="unsupported field: parent_meta"):
        EncoderPCMemory().load_state_dict(with_legacy_meta)


def test_loader_rejects_wrong_tensor_dtype_and_out_of_range_metadata() -> None:
    wrong_dtype = _ready_memory().state_dict()
    wrong_dtype["route"]["route_keys"] = wrong_dtype["route"]["route_keys"].float()
    with pytest.raises(ValueError, match="route.route_keys must be float16"):
        EncoderPCMemory().load_state_dict(wrong_dtype)

    wrong_pointer = _ready_memory().state_dict()
    wrong_pointer["parent"]["child_ptr"][0] = 100
    with pytest.raises(ValueError, match="child_ptr"):
        EncoderPCMemory().load_state_dict(wrong_pointer)

    wrong_image = _ready_memory().state_dict()
    wrong_image["child"]["image_index"][0] = 10
    with pytest.raises(ValueError, match="child.image_index"):
        EncoderPCMemory().load_state_dict(wrong_image)


def test_compatibility_checks_static_producer_and_labeled_split_contracts() -> None:
    memory = _ready_memory()
    expected = _compat()
    assert memory.validate_compat(expected)
    assert memory.validate_compatibility(expected)

    wrong_producer = _compat(producer="different-adapter")
    ok, reason = memory.validate_compat(wrong_producer)
    assert not ok
    assert reason == "compat_mismatch:producer_fingerprint"
    assert memory.validate_compat(wrong_producer, require_producer_match=False)

    wrong_split = _compat(split="different-labeled-split")
    assert memory.validate_compat(wrong_split).reason == (
        "compat_mismatch:labeled_split_fingerprint"
    )
    assert memory.validate_compat(wrong_split, require_split_match=False)

    wrong_architecture = dict(expected, architecture="DINO_SCOD_BGFBR_PC_HBM")
    assert memory.validate_compat(wrong_architecture).reason == (
        "compat_mismatch:architecture"
    )
    with pytest.raises(RuntimeError, match="rebuild memory from the labeled split"):
        memory.assert_compatible(wrong_architecture)


def test_complete_meta_and_value_reliability_contract_are_required() -> None:
    memory = EncoderPCMemory()
    memory.append(_entry())
    incomplete = dict(_compat())
    incomplete.pop("producer_fingerprint")
    with pytest.raises(ValueError, match="producer_fingerprint"):
        memory.finalize(compat_meta=incomplete)

    entry = _entry()
    entry["parent"]["values"][:, 7] = 0.0
    memory = EncoderPCMemory()
    memory.append(entry)
    with pytest.raises(ValueError, match=r"values\[:, 7\]"):
        memory.finalize(compat_meta=_compat())
