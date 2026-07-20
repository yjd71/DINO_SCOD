from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from configs.pc_hbm_dino_config import EncoderPCHBMConfig
from Model.PC_HBM.encoder.teacher_pseudo_refiner import (
    EncoderRefinerEvidence,
    RefinerLossWeights,
    TeacherPseudoLabelRefiner,
    teacher_pseudo_refiner_labeled_loss,
)


def _encoder_evidence(
    batch_size: int,
    *,
    requires_grad: bool = False,
) -> dict[str, torch.Tensor]:
    def tensor(channels: int) -> torch.Tensor:
        return torch.rand(
            batch_size,
            channels,
            28,
            28,
            requires_grad=requires_grad,
        )

    return {
        "verified_evidence": tensor(128),
        "boundary_probability": tensor(1),
        "pc_gate": tensor(1),
        "contradiction": tensor(1),
        "semantic_support": tensor(1),
        "detail_support": tensor(1),
        "valid_map": torch.ones(
            batch_size, 1, 28, 28, requires_grad=requires_grad
        ),
        "route_confidence": torch.full(
            (batch_size,), 0.8, requires_grad=requires_grad
        ),
    }


def test_zero_initialized_refiner_has_identity_candidates_and_contract() -> None:
    torch.manual_seed(7)
    refiner = TeacherPseudoLabelRefiner()
    z_core = torch.randn(2, 1, 98, 98)
    decoder_feature = torch.randn(2, 128, 98, 98)

    output = refiner(z_core, decoder_feature, _encoder_evidence(2), epoch=21)

    assert output["candidates"].shape == (2, 4, 98, 98)
    assert output["candidate_probabilities"].shape == (2, 4, 98, 98)
    assert output["pi"].shape == (2, 4, 98, 98)
    assert output["branch_quality"].shape == (2, 4, 98, 98)
    assert output["encoder_evidence_98"].shape == (2, 128, 98, 98)
    assert output["mixture_entropy"].shape == (2, 1, 98, 98)
    assert torch.all(output["suppression"] >= 0.0)
    for index, name in enumerate(
        ("z_keep", "z_residual", "z_deformation", "z_suppress")
    ):
        assert torch.equal(output[name], z_core)
        assert torch.equal(
            output["candidate_probabilities"][:, index : index + 1],
            torch.sigmoid(z_core),
        )
    assert torch.equal(output["z_pseudo_refined"], z_core)
    assert torch.equal(output["p_pseudo_refined"], torch.sigmoid(z_core))
    assert torch.equal(
        refiner.mixture_head.bias.detach(),
        torch.tensor([1.0, -0.5, -0.5, -0.5]),
    )
    for head in (
        refiner.correction_mask_head,
        refiner.residual_head,
        refiner.offset_head,
        refiner.suppress_head,
        refiner.mixture_head,
        refiner.quality_head,
    ):
        assert torch.count_nonzero(head.weight).item() == 0
    for head in (
        refiner.correction_mask_head,
        refiner.residual_head,
        refiner.offset_head,
        refiner.suppress_head,
        refiner.quality_head,
    ):
        assert torch.count_nonzero(head.bias).item() == 0


def test_refiner_owns_detach_boundary_for_all_three_inputs() -> None:
    torch.manual_seed(11)
    refiner = TeacherPseudoLabelRefiner()
    base = torch.linspace(-2.0, 2.0, 98 * 98).reshape(1, 1, 98, 98)
    z_core = base.clone().requires_grad_()
    decoder_feature = torch.randn(1, 128, 98, 98, requires_grad=True)
    evidence = _encoder_evidence(1, requires_grad=True)
    output = refiner(z_core, decoder_feature, evidence, epoch=25)
    gt = (torch.rand(1, 1, 98, 98) > 0.5).float()

    loss, terms = teacher_pseudo_refiner_labeled_loss(output, gt)
    loss.backward()

    assert z_core.grad is None
    assert decoder_feature.grad is None
    assert all(tensor.grad is None for tensor in evidence.values())
    refiner_gradients = [
        parameter.grad
        for parameter in refiner.parameters()
        if parameter.grad is not None
    ]
    assert refiner_gradients
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in refiner_gradients)
    assert set(terms) == {
        "L_refined_final",
        "L_mix_oracle",
        "L_branch",
        "L_quality",
        "L_usage",
        "L_refiner_reg",
        "L_refiner_total",
    }
    assert all(not value.requires_grad for value in terms.values())


def test_labeled_loss_treats_refined_output_as_probability(monkeypatch) -> None:
    refiner = TeacherPseudoLabelRefiner()
    z_core = torch.randn(1, 1, 98, 98)
    output = refiner(z_core, torch.randn(1, 128, 98, 98), _encoder_evidence(1))
    gt = torch.rand(1, 1, 98, 98)
    config = SimpleNamespace(
        lambda_refined_final=1.0,
        lambda_mix_oracle=0.0,
        lambda_branch=0.0,
        lambda_quality=0.0,
        lambda_usage=0.0,
        lambda_refiner_reg=0.0,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("probabilities must not reach BCEWithLogits")

    monkeypatch.setattr(F, "binary_cross_entropy_with_logits", forbidden)
    loss, terms = teacher_pseudo_refiner_labeled_loss(output, gt, config)
    expected = F.binary_cross_entropy(
        output["p_pseudo_refined"].clamp(1.0e-6, 1.0 - 1.0e-6), gt
    )

    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(terms["L_refined_final"], expected)


def test_evidence_contract_rejects_missing_or_wrong_shape() -> None:
    evidence = _encoder_evidence(1)
    del evidence["detail_support"]
    with pytest.raises(KeyError, match="detail_support"):
        EncoderRefinerEvidence.from_mapping(evidence)

    evidence = _encoder_evidence(1)
    evidence["verified_evidence"] = torch.randn(1, 127, 28, 28)
    contract = EncoderRefinerEvidence.from_mapping(evidence)
    with pytest.raises(ValueError, match="verified_evidence"):
        contract.validate(batch_size=1)

    evidence = _encoder_evidence(1)
    evidence["verified_evidence"] = torch.randn(1, 128, 27, 28)
    evidence["boundary_probability"] = torch.rand(1, 1, 27, 28)
    evidence["pc_gate"] = torch.rand(1, 1, 27, 28)
    evidence["contradiction"] = torch.rand(1, 1, 27, 28)
    evidence["semantic_support"] = torch.rand(1, 1, 27, 28)
    evidence["detail_support"] = torch.rand(1, 1, 27, 28)
    evidence["valid_map"] = torch.ones(1, 1, 27, 28)
    contract = EncoderRefinerEvidence.from_mapping(evidence)
    with pytest.raises(ValueError, match="fixed 28x28"):
        contract.validate(batch_size=1)


def test_encoder_config_exposes_canonical_refiner_weight_aliases() -> None:
    config = EncoderPCHBMConfig(
        lambda_refiner_final=1.75,
        lambda_reg=0.125,
    )

    weights = RefinerLossWeights.from_config(config)

    assert config.lambda_refined_final == pytest.approx(1.75)
    assert config.lambda_refiner_reg == pytest.approx(0.125)
    assert weights.refined_final == pytest.approx(1.75)
    assert weights.regularization == pytest.approx(0.125)


def test_conflicting_refiner_weight_aliases_fail_fast() -> None:
    config = SimpleNamespace(
        lambda_refined_final=1.0,
        lambda_refiner_final=2.0,
        lambda_mix_oracle=0.10,
        lambda_branch=0.10,
        lambda_quality=0.025,
        lambda_usage=0.01,
        lambda_refiner_reg=0.02,
        lambda_reg=0.02,
    )

    with pytest.raises(ValueError, match="conflicting refiner weight aliases"):
        RefinerLossWeights.from_config(config)


def test_non_default_encoder_weights_control_the_total_refiner_loss() -> None:
    config = EncoderPCHBMConfig(
        lambda_refiner_final=1.7,
        lambda_mix_oracle=0.21,
        lambda_branch=0.32,
        lambda_quality=0.043,
        lambda_usage=0.054,
        lambda_reg=0.065,
    )
    refiner = TeacherPseudoLabelRefiner(config)
    output = refiner(
        torch.randn(1, 1, 98, 98),
        torch.randn(1, 128, 98, 98),
        _encoder_evidence(1),
        epoch=23,
    )
    gt = torch.randint(0, 2, (1, 1, 98, 98)).float()

    total, terms = teacher_pseudo_refiner_labeled_loss(output, gt, config)
    expected = (
        1.7 * terms["L_refined_final"]
        + 0.21 * terms["L_mix_oracle"]
        + 0.32 * terms["L_branch"]
        + 0.043 * terms["L_quality"]
        + 0.054 * terms["L_usage"]
        + 0.065 * terms["L_refiner_reg"]
    )

    torch.testing.assert_close(total.detach(), expected)


def test_subtractive_suppression_is_nonnegative_without_zero_init_deadlock() -> None:
    refiner = TeacherPseudoLabelRefiner()
    output = refiner(
        torch.randn(1, 1, 98, 98),
        torch.randn(1, 128, 98, 98),
        _encoder_evidence(1),
    )

    assert torch.equal(output["suppression"], torch.zeros_like(output["suppression"]))
    output["z_suppress"].mean().backward()
    gradient = refiner.suppress_head.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient) > 0

    with torch.no_grad():
        refiner.suppress_head.weight.normal_(mean=0.0, std=0.1)
        refiner.suppress_head.bias.fill_(-0.25)
    output = refiner(
        torch.randn(1, 1, 98, 98),
        torch.randn(1, 128, 98, 98),
        _encoder_evidence(1),
    )
    assert torch.all(output["suppression"] >= 0.0)
