from __future__ import annotations

import torch

from Model.PC_HBM.fusion import P3GatedResidual
from Model.PC_HBM.retrieval import PairVerifier


def _retrieval(valid: torch.Tensor) -> dict[str, torch.Tensor]:
    query_count, _, topk = valid.shape
    dim = 4
    parent = torch.zeros(query_count, 2, topk, dim)
    child = torch.zeros_like(parent)
    parent[:, 0, :, 0] = 1.0
    child[:, 0, :, 0] = 1.0
    parent[:, 1, :, 1] = 1.0
    child[:, 1, :, 1] = 1.0
    return {
        "parent_keys": parent,
        "paired_p2_keys": child,
        "valid": valid,
    }


def test_pair_verifier_has_one_scalar_and_fp32_binary_evidence() -> None:
    verifier = PairVerifier(
        dim=4,
        tau_parent=1.0,
        tau_child=1.0,
        child_mix_init_logit=0.0,
    )
    valid = torch.ones(1, 2, 1, dtype=torch.bool)
    result = verifier(
        q3=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        q_child=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        retrieval=_retrieval(valid),
        query_score=torch.full((1, 1), 0.25),
    )

    parameters = list(verifier.named_parameters())
    assert [(name, value.numel()) for name, value in parameters] == [
        ("raw_child_mix", 1)
    ]
    torch.testing.assert_close(result["beta"], torch.tensor(0.5))
    assert result["pair_logits"].dtype == torch.float32
    assert result["region_prob"][0, 0] > result["region_prob"][0, 1]
    assert torch.count_nonzero(result["candidate_entropy"]) == 0
    assert result["query_valid"].tolist() == [True]
    assert torch.isfinite(result["correction"]).all()
    expected_context = (
        result["region_prob"].unsqueeze(-1) * result["region_context"]
    ).sum(dim=1)
    torch.testing.assert_close(result["memory_context"], expected_context)
    torch.testing.assert_close(
        result["correction"],
        expected_context - torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    expected_margin = (
        result["region_prob"][:, 0] - result["region_prob"][:, 1]
    ).abs()
    torch.testing.assert_close(result["region_margin"], expected_margin)
    torch.testing.assert_close(
        result["memory_confidence"], expected_margin[:, None]
    )
    torch.testing.assert_close(
        result["gate"], 0.25 * result["memory_confidence"]
    )


def test_pair_verifier_confidence_uses_probability_weighted_region_entropy() -> None:
    verifier = PairVerifier(
        dim=4,
        tau_parent=1.0,
        tau_child=1.0,
    )
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    retrieval = _retrieval(valid)
    retrieval["parent_keys"][0, 1, 1] = torch.tensor(
        [-1.0, 0.0, 0.0, 0.0]
    )
    retrieval["paired_p2_keys"][0, 1, 1] = torch.tensor(
        [-1.0, 0.0, 0.0, 0.0]
    )
    result = verifier(
        q3=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        q_child=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        retrieval=retrieval,
        query_score=torch.full((1, 1), 0.4),
    )

    attention = result["pair_weight"]
    region_entropy = -(
        attention * attention.clamp_min(1.0e-12).log()
    ).sum(dim=-1) / torch.log(torch.tensor(2.0))
    expected_entropy = (
        result["region_prob"] * region_entropy
    ).sum(dim=-1)
    expected_margin = (
        result["region_prob"][:, 0] - result["region_prob"][:, 1]
    ).abs()
    expected_confidence = (1.0 - expected_entropy) * expected_margin
    torch.testing.assert_close(
        result["candidate_entropy"], expected_entropy
    )
    torch.testing.assert_close(result["region_margin"], expected_margin)
    torch.testing.assert_close(
        result["memory_confidence"], expected_confidence[:, None]
    )
    torch.testing.assert_close(
        result["gate"], 0.4 * expected_confidence[:, None]
    )
    assert bool((result["candidate_entropy"] >= 0).all())
    assert bool((result["candidate_entropy"] <= 1).all())
    assert bool((result["memory_confidence"] >= 0).all())
    assert bool((result["memory_confidence"] <= 1).all())


def test_pair_verifier_fully_invalid_query_is_exact_zero_without_nan() -> None:
    verifier = PairVerifier(dim=4)
    valid = torch.zeros(2, 2, 4, dtype=torch.bool)
    result = verifier(
        q3=torch.randn(2, 4),
        q_child=torch.randn(2, 4),
        retrieval=_retrieval(valid),
        query_score=torch.ones(2, 1),
    )

    assert not result["query_valid"].any()
    for key in (
        "pair_logits",
        "region_prob",
        "region_context",
        "memory_context",
        "region_margin",
        "memory_confidence",
        "gate",
        "correction",
    ):
        assert torch.isfinite(result[key]).all(), key
        assert torch.count_nonzero(result[key]) == 0, key


def test_retrieval_and_verifier_accept_zero_queries() -> None:
    query = torch.empty(0, 4)
    valid = torch.empty(0, 2, 4, dtype=torch.bool)
    result = PairVerifier(dim=4)(
        query,
        query,
        _retrieval(valid),
        query_score=torch.empty(0, 1),
    )
    assert result["pair_logits"].shape == (0, 2)
    assert result["region_prob"].shape == (0, 2)
    assert result["correction"].shape == (0, 4)
    assert result["query_valid"].shape == (0,)


def test_single_residual_is_zero_initialized_and_never_touches_nonqueries() -> None:
    module = P3GatedResidual(dim=4, p3_ch=4)
    p3 = torch.randn(1, 4, 3, 3)
    batch_ids = torch.tensor([0])
    flat_indices = torch.tensor([4])
    correction = torch.ones(1, 4)
    valid = torch.ones(1, dtype=torch.bool)

    corrected, delta = module(
        p3, batch_ids, flat_indices, correction, torch.ones(1, 1), valid
    )
    assert torch.equal(corrected, p3)
    assert delta.shape == (1, 4)
    assert torch.count_nonzero(delta) == 0

    with torch.no_grad():
        module.out.weight.copy_(torch.eye(4))
        module.out.bias.zero_()
    corrected, delta = module(
        p3, batch_ids, flat_indices, correction, torch.ones(1, 1), valid
    )
    nonquery = torch.ones(3, 3, dtype=torch.bool)
    nonquery[1, 1] = False
    assert torch.equal(corrected[0, :, nonquery], p3[0, :, nonquery])
    assert delta.shape == (1, 4)
    torch.testing.assert_close(delta[0], torch.ones(4))
