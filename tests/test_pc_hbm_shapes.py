"""Pure-tensor contracts for DINO PC-HBM fusion/refinement leaf modules."""

from __future__ import annotations

import torch

from Model.PC_HBM.fusion import (
    HypothesisTokenBuilder,
    P3GatedResidual,
    PCHCA,
    PCTokenDecoder,
    QueryStateBuilder,
    StructuredGateMLP,
    pc_scatter,
)
from Model.PC_HBM.refinement import (
    AdaptiveMixtureHead,
    BoundaryQueryHead,
    P1PixelRefinementAttention,
    P2BoundaryRetargetAttention,
)


def _parent_child_inputs() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    torch.manual_seed(7)
    query_count, candidate_count, dim = 3, 4, 128
    valid = torch.tensor(
        [
            [True, True, False, False],
            [False, False, False, False],
            [True, False, True, False],
        ]
    )
    parent = {
        "top_parent_keys": torch.randn(query_count, candidate_count, dim),
        "top_parent_values": torch.randn(query_count, candidate_count, 8),
        "top_parent_geo": torch.randn(query_count, candidate_count, 6),
        "top_parent_scores": torch.randn(query_count, candidate_count),
        "top_parent_valid": valid,
    }
    child = {
        "K_child_top": torch.randn(query_count, candidate_count, dim),
        "G2_child_top": torch.randn(query_count, candidate_count, 6),
        "S_child": torch.randn(query_count, candidate_count),
        "S_geo": torch.randn(query_count, candidate_count),
        "prior_bias": torch.randn(query_count, candidate_count),
    }
    return parent, child


def test_fusion_valid_mask_and_all_invalid_identity() -> None:
    parent, child = _parent_child_inputs()
    valid = parent["top_parent_valid"]
    query_valid = valid.any(dim=1)
    hypothesis = HypothesisTokenBuilder()(parent, child)
    assert hypothesis.shape == (3, 4, 128)
    assert torch.count_nonzero(hypothesis[~valid]) == 0

    query_state = QueryStateBuilder()(
        torch.randn(3, 128),
        torch.randn(3, 128),
        torch.randn(3, 128),
        torch.rand(3, 1),
        torch.rand(3),
        query_valid=query_valid,
    )
    hca = PCHCA()
    query_new, attention = hca(
        query_state,
        hypothesis,
        child["prior_bias"],
        torch.randn(3, 128),
        mask=valid,
        query_valid=query_valid,
    )
    assert query_new.shape == (3, 128)
    assert attention.shape == (3, 4)
    assert torch.equal(query_new[1], query_state[1])
    assert torch.count_nonzero(attention[1]) == 0
    assert torch.count_nonzero(attention[~valid]) == 0

    token_aux = PCTokenDecoder()(
        query_new,
        attention,
        parent,
        child,
        top_parent_valid=valid,
        query_valid=query_valid,
    )
    for key in (
        "E_attn",
        "G_attn",
        "G_child_attn",
        "M_pc_token",
        "O_pc_token",
        "Z3_token",
    ):
        assert torch.count_nonzero(token_aux[key][1]) == 0

    gate = StructuredGateMLP()(
        torch.rand(3, 1),
        torch.rand(3, 1),
        torch.rand(3, 1),
        torch.rand(3),
        torch.rand(3),
        child["S_child"],
        child["S_geo"],
        top_parent_valid=valid,
        query_valid=query_valid,
    )
    assert gate.shape == (3, 1)
    assert gate[1].item() == 0.0

    batch_ids = torch.tensor([0, 0, 0])
    flat_indices = torch.tensor([0, 1, 2])
    maps = pc_scatter(
        1,
        28,
        28,
        batch_ids,
        flat_indices,
        token_aux,
        gate,
        torch.rand(3, 1),
        query_valid=query_valid,
    )
    assert maps["Z3_map"].shape == (1, 128, 28, 28)
    assert maps["valid3_map"][0, 0, 1, 0].item() == 0.0

    p3 = torch.randn(1, 128, 28, 28)
    p3_corr, delta = P3GatedResidual()(
        p3,
        batch_ids,
        flat_indices,
        query_new,
        gate=1.0,
        gate_pc=gate,
        query_valid=query_valid,
    )
    assert torch.equal(p3_corr, p3)
    assert torch.count_nonzero(delta) == 0


def test_boundary_query_all_invalid_is_empty() -> None:
    head = BoundaryQueryHead(5, min_tokens=4, max_tokens=8)
    score, indices = head(
        torch.randn(2, 5, 28, 28),
        valid_mask=torch.zeros(2, 1, 28, 28, dtype=torch.bool),
    )
    assert torch.count_nonzero(score) == 0
    assert indices["batch_ids"].numel() == 0
    assert indices["flat_indices"].numel() == 0


def _pc_maps(batch_size: int = 1) -> dict[str, torch.Tensor]:
    return {
        "Z3_map": torch.randn(batch_size, 128, 28, 28),
        "E_attn_map": torch.randn(batch_size, 8, 28, 28),
        "G_attn_map": torch.randn(batch_size, 6, 28, 28),
        "M_pc_map": torch.randn(batch_size, 1, 28, 28),
        "gate_pc_map": torch.rand(batch_size, 1, 28, 28),
        "C23_map": torch.rand(batch_size, 1, 28, 28),
        "O_pc_map": torch.randn(batch_size, 2, 28, 28),
        "valid3_map": torch.ones(batch_size, 1, 28, 28),
    }


def test_p2_p1_and_mixture_shapes_and_zero_init_identity() -> None:
    torch.manual_seed(11)
    p2 = torch.randn(1, 128, 28, 28)
    p2_module = P2BoundaryRetargetAttention(
        p2_ch=128, min_tokens=4, max_tokens=8
    ).eval()
    p2_aux = p2_module(p2, torch.rand(1, 1, 28, 28), _pc_maps())
    assert torch.equal(p2_aux["p2_refined"], p2)
    assert p2_aux["F2_ref_map"].shape == (1, 128, 28, 28)
    assert p2_aux["B2_refined_map"].shape == (1, 1, 28, 28)
    assert p2_aux["O2_refined_map"].shape == (1, 2, 28, 28)

    p1 = torch.randn(1, 128, 98, 98)
    z_main = torch.randn(1, 1, 98, 98)
    p1_module = P1PixelRefinementAttention(
        p1_ch=128, min_tokens=8, max_tokens=12
    ).eval()
    p1_aux = p1_module(p1, z_main, p2_aux)
    assert p1_aux["G1_map"].shape == (1, 1, 98, 98)
    assert p1_aux["R1_map"].shape == (1, 1, 98, 98)
    assert p1_aux["O1_map"].shape == (1, 2, 98, 98)
    assert p1_aux["R_sup_map"].shape == (1, 1, 98, 98)
    assert torch.count_nonzero(p1_aux["R1_map"]) == 0
    assert torch.count_nonzero(p1_aux["O1_map"]) == 0
    assert torch.count_nonzero(p1_aux["R_sup_map"]) == 0

    mixture = AdaptiveMixtureHead(use_branch_dropout=False).eval()
    mix_aux = mixture(z_main, p1_aux, _pc_maps())
    assert mix_aux["pi"].shape == (1, 4, 98, 98)
    assert mix_aux["z_final"].shape == (1, 1, 98, 98)
    for key in ("z_keep", "z_res", "z_def", "z_sup", "z_final"):
        assert torch.equal(mix_aux[key], z_main), key
    assert torch.equal(mix_aux["p_final"], torch.sigmoid(z_main))


def test_refinement_all_invalid_is_identity() -> None:
    p2 = torch.randn(1, 128, 28, 28)
    maps = _pc_maps()
    maps["valid3_map"].zero_()
    p2_aux = P2BoundaryRetargetAttention(
        p2_ch=128, min_tokens=4, max_tokens=8
    )(p2, torch.rand(1, 1, 28, 28), maps)
    assert torch.equal(p2_aux["p2_refined"], p2)
    assert torch.count_nonzero(p2_aux["valid2_map"]) == 0

    p1 = torch.randn(1, 128, 98, 98)
    z_main = torch.randn(1, 1, 98, 98)
    p1_aux = P1PixelRefinementAttention(
        p1_ch=128, min_tokens=8, max_tokens=12
    )(p1, z_main, p2_aux)
    assert torch.count_nonzero(p1_aux["valid1_map"]) == 0
    mix_aux = AdaptiveMixtureHead(use_branch_dropout=False)(z_main, p1_aux, maps)
    assert torch.equal(mix_aux["z_final"], z_main)

