from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from Model.PC_HBM.encoder import (
    DinoFeatureBundle,
    EncoderPCHBMAdapter,
    EncoderPCMemory,
    EncoderPCStageFlags,
    build_encoder_memory_compat_meta,
)
from Model.PC_HBM.encoder.route_context_adapter import EncoderStructuredGate


def _bundle(batch_size: int = 1) -> DinoFeatureBundle:
    torch.manual_seed(41)
    patches = tuple(torch.randn(batch_size, 784, 768) for _ in range(4))
    cls = tuple(torch.randn(batch_size, 768) for _ in range(4))
    return DinoFeatureBundle(patches, cls).validate()


def _memory(image_count: int = 2, parents_per_image: int = 20) -> EncoderPCMemory:
    torch.manual_seed(43)
    parent_count = image_count * parents_per_image
    image_index = torch.arange(image_count).repeat_interleave(parents_per_image)
    reliability = torch.linspace(0.55, 0.95, parent_count)
    values = torch.randn(parent_count, 8)
    values[:, 7] = reliability
    geometry = torch.zeros(parent_count, 6)
    geometry[:, 1] = 1.0
    geometry[:, 5] = reliability
    memory = EncoderPCMemory()
    memory.append(
        {
            "source": "labeled_only",
            "route": {
                "route_keys": torch.randn(image_count, 128),
                "cls4_keys": torch.randn(image_count, 128),
                "f4_global_keys": torch.randn(image_count, 128),
                "f3_boundary_keys": torch.randn(image_count, 128),
                "image_ids": [f"image-{index}" for index in range(image_count)],
            },
            "parent": {
                "f3_parent_keys": torch.randn(parent_count, 128),
                "values": values,
                "geometry": geometry,
                "child_ptr": torch.arange(parent_count),
                "image_index": image_index,
                "region_id": torch.arange(parent_count) % 4,
                "flat_index": torch.arange(parent_count) % 784,
                "reliability": reliability,
            },
            "child": {
                "f2_child_keys": torch.randn(parent_count, 128),
                "f1_detail_keys": torch.randn(parent_count, 128),
                "geometry": geometry.clone(),
                "image_index": image_index.clone(),
                "flat_index": torch.arange(parent_count) % 784,
            },
        }
    )
    memory.finalize(
        compat_meta=build_encoder_memory_compat_meta(
            dino_weight_fingerprint="identity-dino-sha256",
            producer_fingerprint="identity-adapter-sha256",
            labeled_split_fingerprint="identity-split-sha256",
        )
    )
    return memory


def test_off_is_exact_identity_and_executes_no_adapter_submodule() -> None:
    adapter = EncoderPCHBMAdapter()
    bundle = _bundle()
    adapter.bootstrap.forward = Mock(side_effect=AssertionError("bootstrap was called"))

    output = adapter(bundle, mode="off")

    assert output.features is bundle.patch_tokens
    assert all(torch.equal(actual, expected) for actual, expected in zip(output.features, bundle.patch_tokens))
    adapter.bootstrap.forward.assert_not_called()


def test_parent_only_keeps_all_four_features_identical() -> None:
    adapter = EncoderPCHBMAdapter().eval()
    bundle = _bundle()

    output = adapter(bundle, memory=_memory(), mode="parent_only")

    assert all(torch.equal(actual, expected) for actual, expected in zip(output.features, bundle.patch_tokens))
    assert output.aux["parent"]["top_parent_keys"].shape[-2:] == (16, 128)


def test_initial_full_adapter_is_exact_feature_identity_with_structured_maps() -> None:
    adapter = EncoderPCHBMAdapter().train()
    bundle = _bundle()

    output = adapter(
        bundle,
        memory=_memory(),
        mode="full",
        stage=EncoderPCStageFlags(enable_f4_f3=True, f4_f3_progress=0.2),
    )

    assert all(torch.equal(actual, expected) for actual, expected in zip(output.features, bundle.patch_tokens))
    assert torch.count_nonzero(output.aux["injection"].f3_delta) == 0
    assert torch.count_nonzero(output.aux["injection"].f4_delta) == 0
    assert output.aux["route_context"].verified_f3_map.shape == (1, 128, 28, 28)
    assert output.aux["C23_map"].shape == (1, 1, 28, 28)
    evidence = output.aux["refiner_evidence"]
    assert evidence["verified_evidence"].shape == (1, 128, 28, 28)
    assert evidence["boundary_probability"].shape == (1, 1, 28, 28)
    for name in (
        "pc_gate",
        "contradiction",
        "semantic_support",
        "detail_support",
        "valid_map",
    ):
        assert evidence[name].shape == (1, 1, 28, 28)
    assert evidence["route_confidence"].shape == (1,)
    assert output.aux["injection"].f4_strength.item() > 0.0
    assert output.aux["injection"].f3_strength.item() > 0.0


def test_zero_restore_receives_gradient_without_alpha_deadlock() -> None:
    adapter = EncoderPCHBMAdapter().train()
    output = adapter(
        _bundle(),
        memory=_memory(),
        mode="full",
        stage=EncoderPCStageFlags(enable_f4_f3=True, f4_f3_progress=1.0),
    )

    sum(feature.square().mean() for feature in output.features[2:]).backward()

    assert adapter.injector.restore4.weight.grad is not None
    assert adapter.injector.restore3.weight.grad is not None
    assert torch.count_nonzero(adapter.injector.restore4.weight.grad) > 0
    assert torch.count_nonzero(adapter.injector.restore3.weight.grad) > 0
    assert adapter.injector.alpha4.item() == pytest.approx(1.0)
    assert adapter.injector.alpha3.item() == pytest.approx(1.0)


def test_all_invalid_route_stays_identity_after_restore_bias_learning() -> None:
    adapter = EncoderPCHBMAdapter().train()
    bundle = _bundle()
    with torch.no_grad():
        adapter.injector.restore4.bias.fill_(0.25)
        adapter.injector.restore3.bias.fill_(0.25)

    output = adapter(
        bundle,
        memory=_memory(image_count=1),
        mode="full",
        query_image_ids=["image-0"],
        stage=EncoderPCStageFlags(enable_f4_f3=True, f4_f3_progress=1.0),
    )

    assert not bool(output.aux["route"]["route_valid"].item())
    assert not bool(output.aux["pc_active"])
    assert all(torch.equal(actual, expected) for actual, expected in zip(output.features, bundle.patch_tokens))
    sum(feature.square().mean() for feature in output.features[2:]).backward()
    assert torch.count_nonzero(adapter.injector.restore4.bias.grad) == 0
    assert torch.count_nonzero(adapter.injector.restore3.bias.grad) == 0


def test_f4_injection_scales_with_route_confidence() -> None:
    adapter = EncoderPCHBMAdapter()
    injector = adapter.injector
    with torch.no_grad():
        injector.restore4.weight.fill_(1.0 / 128.0)
    tokens = torch.zeros(2, 784, 768)
    evidence = torch.ones(2, 128)
    output = injector(
        f3_tokens=tokens,
        f4_tokens=tokens,
        route_evidence=evidence,
        route_confidence=torch.tensor([0.2, 1.0]),
        route_valid=torch.ones(2, dtype=torch.bool),
        verified_f3_map=torch.zeros(2, 128, 28, 28),
        f3_gate_map=torch.zeros(2, 1, 28, 28),
        progress=1.0,
    )

    low = output.f4_delta[0].abs().mean()
    high = output.f4_delta[1].abs().mean()
    assert torch.allclose(high, low * 5.0)


def test_f4_route_delta_is_conditioned_on_each_online_token() -> None:
    adapter = EncoderPCHBMAdapter()
    injector = adapter.injector
    with torch.no_grad():
        injector.restore4.weight.fill_(1.0 / 128.0)
    f4 = torch.zeros(1, 784, 768)
    f4[:, 1] = 1.0
    output = injector(
        f3_tokens=torch.zeros_like(f4),
        f4_tokens=f4,
        route_evidence=torch.ones(1, 128),
        route_confidence=torch.ones(1),
        route_valid=torch.ones(1, dtype=torch.bool),
        verified_f3_map=torch.zeros(1, 128, 28, 28),
        f3_gate_map=torch.zeros(1, 1, 28, 28),
        progress=1.0,
    )

    assert not torch.equal(output.f4_delta[:, 0], output.f4_delta[:, 1])


def test_encoder_gate_is_isolated_and_contains_all_structured_groups() -> None:
    gate = EncoderStructuredGate()

    assert gate.gamma_gate.item() == 0.0
    assert gate.gate_interaction_mlp[0].in_features == 15
    assert hasattr(gate, "semantic_nam")
    assert hasattr(gate, "detail_nam")
    assert hasattr(gate, "geometry_nam")
    valid = torch.zeros(2, 16, dtype=torch.bool)
    result = gate(
        route_confidence=torch.ones(2),
        contradiction=torch.zeros(2, 1),
        uncertainty=torch.zeros(2, 1),
        parent_entropy=torch.zeros(2),
        child_entropy=torch.zeros(2),
        parent_scores=torch.zeros(2, 16),
        semantic_scores=torch.zeros(2, 16),
        detail_scores=torch.zeros(2, 16),
        geometry_scores=torch.zeros(2, 16),
        boundary_confidence=torch.ones(2, 1),
        valid=valid,
        query_valid=torch.zeros(2, dtype=torch.bool),
    )
    assert torch.equal(result, torch.zeros_like(result))


def test_full_mode_requires_ready_v3_memory() -> None:
    adapter = EncoderPCHBMAdapter().train()
    with pytest.raises(RuntimeError, match="schema-v3 memory"):
        adapter(_bundle(), mode="full")


def test_memory_features_use_raw_bundle_and_fixed_128d_contract() -> None:
    adapter = EncoderPCHBMAdapter().eval()
    features = adapter.forward_memory_features(_bundle(batch_size=2))

    assert set(features) == {
        "route_keys",
        "cls4_keys",
        "f4_global_keys",
        "f3_boundary_keys",
        "f3_parent_keys",
        "f2_child_keys",
        "f1_detail_keys",
    }
    for name in ("route_keys", "cls4_keys", "f4_global_keys", "f3_boundary_keys"):
        assert features[name].shape == (2, 128)
    for name in ("f3_parent_keys", "f2_child_keys", "f1_detail_keys"):
        assert features[name].shape == (2, 784, 128)
