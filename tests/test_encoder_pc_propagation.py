from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from configs.pc_hbm_dino_config import EncoderPCHBMConfig
from Model.PC_HBM.encoder import (
    DinoFeatureBundle,
    EncoderLevelPropagation,
    EncoderPCHBMAdapter,
    EncoderPCStageFlags,
    SameGridLocalCrossAttention,
)
from tests.test_encoder_pc_identity import _memory


def _propagation_inputs(*, requires_grad: bool = False) -> dict[str, torch.Tensor]:
    torch.manual_seed(53)
    return {
        "f1_tokens": torch.randn(1, 784, 768),
        "f2_tokens": torch.randn(1, 784, 768),
        "e1_map": torch.randn(1, 128, 28, 28, requires_grad=requires_grad),
        "e2_map": torch.randn(1, 128, 28, 28, requires_grad=requires_grad),
        "corrected_f3_state": torch.randn(
            1, 128, 28, 28, requires_grad=requires_grad
        ),
        "verified_f2_map": torch.randn(1, 128, 28, 28),
        "verified_f1_map": torch.randn(1, 128, 28, 28),
        "valid2_map": torch.ones(1, 1, 28, 28, dtype=torch.bool),
        "valid1_map": torch.ones(1, 1, 28, 28, dtype=torch.bool),
    }


def _bundle() -> DinoFeatureBundle:
    torch.manual_seed(59)
    return DinoFeatureBundle(
        tuple(torch.randn(1, 784, 768) for _ in range(4)),
        tuple(torch.randn(1, 768) for _ in range(4)),
    ).validate()


def test_initial_f2_f1_propagation_is_exact_identity_and_has_3x3_attention() -> None:
    propagation = EncoderLevelPropagation()
    values = _propagation_inputs()

    output = propagation(**values, progress=0.2)

    assert torch.equal(output.f1_tokens, values["f1_tokens"])
    assert torch.equal(output.f2_tokens, values["f2_tokens"])
    assert output.f1_attention.shape == (1, 784, 9)
    assert output.f2_attention.shape == (1, 784, 9)
    assert torch.count_nonzero(output.f1_delta) == 0
    assert torch.count_nonzero(output.f2_delta) == 0


def test_all_invalid_propagation_stays_identity_after_bias_learning() -> None:
    propagation = EncoderLevelPropagation()
    values = _propagation_inputs()
    values["valid1_map"].zero_()
    values["valid2_map"].zero_()
    with torch.no_grad():
        propagation.restore1.bias.fill_(0.4)
        propagation.restore2.bias.fill_(0.4)

    output = propagation(**values, progress=1.0)

    assert torch.equal(output.f1_tokens, values["f1_tokens"])
    assert torch.equal(output.f2_tokens, values["f2_tokens"])
    assert torch.count_nonzero(output.f1_attention) == 0
    assert torch.count_nonzero(output.f2_attention) == 0
    assert not bool(output.valid1_map.any())
    assert not bool(output.valid2_map.any())
    (output.f1_tokens.square().mean() + output.f2_tokens.square().mean()).backward()
    assert torch.count_nonzero(propagation.restore1.bias.grad) == 0
    assert torch.count_nonzero(propagation.restore2.bias.grad) == 0


def test_references_are_detached_between_hierarchy_levels() -> None:
    propagation = EncoderLevelPropagation(
        detach_f3_refs=True,
        detach_f2_refs=True,
    )
    values = _propagation_inputs(requires_grad=True)
    with torch.no_grad():
        propagation.restore1.weight.fill_(1.0 / 128.0)
        propagation.restore2.weight.fill_(1.0 / 128.0)

    output = propagation(**values, progress=1.0)
    output.f2_delta.square().mean().backward(retain_graph=True)
    assert values["corrected_f3_state"].grad is None
    assert values["e2_map"].grad is not None

    values["e2_map"].grad = None
    output.f1_delta.square().mean().backward()
    assert values["e2_map"].grad is None
    assert values["e1_map"].grad is not None


def test_valid_reference_is_masked_and_propagates_to_local_neighborhoods() -> None:
    propagation = EncoderLevelPropagation()
    values = _propagation_inputs()
    values["valid1_map"].zero_()
    values["valid2_map"].zero_()
    values["valid2_map"][:, :, 14, 14] = True

    output = propagation(**values, progress=1.0)

    assert output.valid2_map.sum().item() == 9
    assert output.valid1_map.sum().item() == 25
    outside = output.f2_attention[0, 0]
    assert torch.count_nonzero(outside) == 0


def test_invalid_neighbors_and_padding_cannot_enter_local_attention() -> None:
    attention = SameGridLocalCrossAttention(dim=8, num_heads=1, window_size=3)
    with torch.no_grad():
        for layer in (attention.q, attention.k):
            layer.weight.zero_()
        for layer in (attention.v, attention.out):
            layer.weight.zero_()
            layer.weight[:, :, 0, 0].copy_(torch.eye(8))
    query = torch.zeros(1, 8, 3, 3)
    reference = torch.zeros_like(query)
    reference[:, :, 0, 0] = 1.0
    valid = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
    valid[:, :, 0, 0] = True
    corrupted = reference.clone()
    corrupted[:, :, 0, 1:] = 1.0e4

    clean_state, clean_attention, _ = attention(
        query, reference, torch.zeros_like(query), valid
    )
    corrupt_state, corrupt_attention, _ = attention(
        query, corrupted, torch.zeros_like(query), valid
    )

    assert torch.equal(clean_state, corrupt_state)
    assert torch.equal(clean_attention, corrupt_attention)
    assert clean_attention[0, 0].sum().item() == 1.0


def test_full_adapter_propagates_f2_f1_only_when_stage_enables_it() -> None:
    adapter = EncoderPCHBMAdapter().train()
    bundle = _bundle()
    original_forward = adapter.propagation.forward
    adapter.propagation.forward = Mock(side_effect=AssertionError("propagation called"))
    disabled = adapter(
        bundle,
        memory=_memory(),
        mode="full",
        stage=EncoderPCStageFlags(enable_f2_f1=False, f2_f1_progress=1.0),
    )
    adapter.propagation.forward.assert_not_called()
    adapter.propagation.forward = original_forward
    enabled = adapter(
        bundle,
        memory=_memory(),
        mode="full",
        stage=EncoderPCStageFlags(enable_f2_f1=True, f2_f1_progress=0.2),
    )

    assert torch.equal(disabled.features[0], bundle.patch_tokens[0])
    assert torch.equal(disabled.features[1], bundle.patch_tokens[1])
    assert torch.equal(enabled.features[0], bundle.patch_tokens[0])
    assert torch.equal(enabled.features[1], bundle.patch_tokens[1])
    assert enabled.aux["propagation"].f1_attention.shape == (1, 784, 9)
    assert enabled.aux["propagation"].f2_attention.shape == (1, 784, 9)


def test_restore1_restore2_receive_gradient_at_first_enabled_step() -> None:
    adapter = EncoderPCHBMAdapter().train()
    output = adapter(
        _bundle(),
        memory=_memory(),
        mode="full",
        stage=EncoderPCStageFlags(enable_f2_f1=True, f2_f1_progress=0.2),
    )

    (output.features[0].square().mean() + output.features[1].square().mean()).backward()

    assert torch.count_nonzero(adapter.propagation.restore1.weight.grad) > 0
    assert torch.count_nonzero(adapter.propagation.restore2.weight.grad) > 0


@pytest.mark.parametrize(
    "override",
    [
        {"propagation_window_size": 5},
        {"detach_f3_refs_for_f2": False},
        {"detach_f2_refs_for_f1": False},
        {"attention_heads": 4},
    ],
)
def test_config_rejects_noncanonical_hierarchy_contract(override: dict) -> None:
    with pytest.raises(ValueError):
        EncoderPCHBMConfig(**override)
