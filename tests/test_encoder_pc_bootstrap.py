from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from configs.pc_hbm_dino_config import EncoderPCHBMConfig
from Model.PC_HBM.encoder import DinoFeatureBundle, EncoderBootstrap
from Model.PC_HBM.training.encoder_losses import encoder_bootstrap_loss


def _bundle(batch: int = 2, *, requires_grad: bool = False) -> DinoFeatureBundle:
    patch = tuple(
        torch.randn(batch, 784, 768, requires_grad=requires_grad) for _ in range(4)
    )
    cls = tuple(
        torch.randn(batch, 768, requires_grad=requires_grad) for _ in range(4)
    )
    return DinoFeatureBundle(patch, cls).validate()


def test_encoder_config_locks_v3_contract_and_five_stages():
    config = EncoderPCHBMConfig()

    assert config.pc_placement == "encoder"
    assert config.memory_schema_version == 3
    assert config.stage_for_epoch(1) == "bootstrap"
    assert config.stage_for_epoch(6) == "parent_only"
    assert config.stage_for_epoch(11) == "parent_child_f3"
    assert config.stage_for_epoch(16) == "hierarchical_full"
    assert config.stage_for_epoch(21) == "hierarchical_refiner"
    assert config.stage_progress(11, level="f4_f3") == pytest.approx(0.2)
    assert config.stage_progress(15, level="f4_f3") == 1.0
    assert config.stage_progress(16, level="f2_f1") == pytest.approx(0.2)

    with pytest.raises(ValueError, match="fixed contract"):
        EncoderPCHBMConfig(memory_schema_version=2)


def test_bootstrap_projects_all_levels_and_returns_coarse_boundary_shapes():
    module = EncoderBootstrap().eval()
    output = module(_bundle())

    assert len(output.projected.patch_tokens) == 4
    assert len(output.projected.cls_tokens) == 4
    assert all(tensor.shape == (2, 784, 128) for tensor in output.projected.patch_tokens)
    assert all(tensor.shape == (2, 128) for tensor in output.projected.cls_tokens)
    assert all(tensor.shape == (2, 128, 28, 28) for tensor in output.projected.maps)
    assert output.global_output.fused_map.shape == (2, 128, 28, 28)
    assert output.global_output.coarse_logits.shape == (2, 1, 28, 28)
    assert output.boundary_output.boundary_logits.shape == (2, 1, 28, 28)
    assert output.boundary_output.evidence_channels.shape == (2, 6, 28, 28)
    assert output.boundary_output.selected_indices.shape == (2, 128)
    assert output.boundary_output.selected_mask.all()
    assert torch.equal(
        output.global_output.coarse_probability,
        torch.sigmoid(output.global_output.coarse_logits),
    )
    assert torch.equal(
        output.boundary_output.boundary_probability,
        torch.sigmoid(output.boundary_output.boundary_logits),
    )


def test_patch_and_cls_levels_use_independent_projectors():
    module = EncoderBootstrap()
    patch_weights = [
        projector[0].weight for projector in module.projector.patch_projectors
    ]
    cls_weights = [
        projector[0].weight for projector in module.projector.cls_projectors
    ]

    assert len({weight.data_ptr() for weight in patch_weights + cls_weights}) == 8


def test_bootstrap_losses_use_raw_logits_and_backpropagate_only_modules():
    torch.manual_seed(7)
    module = EncoderBootstrap()
    bundle = _bundle(requires_grad=False)
    output = module(bundle)
    mask = torch.randint(0, 2, (2, 1, 392, 392), dtype=torch.float32)
    boundary = torch.randint(0, 2, (2, 1, 392, 392), dtype=torch.float32)

    losses = encoder_bootstrap_loss(
        coarse_logits=output.global_output.coarse_logits,
        boundary_logits=output.boundary_output.boundary_logits,
        mask_target=mask,
        boundary_target=boundary,
    )
    resized_mask = F.interpolate(mask, (28, 28), mode="nearest")
    resized_boundary = F.interpolate(boundary, (28, 28), mode="nearest")
    expected_coarse = F.binary_cross_entropy_with_logits(
        output.global_output.coarse_logits.float(), resized_mask
    )
    expected_boundary = F.binary_cross_entropy_with_logits(
        output.boundary_output.boundary_logits.float(), resized_boundary
    )
    assert torch.allclose(losses["coarse"], expected_coarse)
    assert torch.allclose(losses["boundary"], expected_boundary)
    assert torch.allclose(
        losses["total"], 0.30 * expected_coarse + 0.10 * expected_boundary
    )

    losses["total"].backward()
    assert module.projector.patch_projectors[0][0].weight.grad is not None
    assert module.global_fusion.coarse_head.weight.grad is not None
    assert module.boundary_query.head[-1].weight.grad is not None
    assert all(not tensor.requires_grad for tensor in bundle.patch_tokens)


def test_bootstrap_loss_has_bce_with_logits_zero_logit_value():
    logits = torch.zeros(1, 1, 28, 28, requires_grad=True)
    target = torch.zeros(1, 1, 28, 28)
    losses = encoder_bootstrap_loss(
        coarse_logits=logits,
        boundary_logits=logits,
        mask_target=target,
        boundary_target=target,
    )
    assert losses["coarse"].item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
