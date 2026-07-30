from __future__ import annotations

import torch

from Model.PC_HBM.retrieval import BalancedParentRetriever, ChildQueryBuilder


def test_balanced_parent_retrieval_keeps_a_fixed_quota_per_region() -> None:
    retriever = BalancedParentRetriever(dim=4, topk_per_region=4)
    query = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    fg = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(10, 1)
    bg = torch.tensor([[0.0, 1.0, 0.0, 0.0]]).repeat(6, 1)
    bank = {
        "p3_keys": torch.cat((fg, bg), dim=0),
        "p2_keys": torch.arange(16 * 4, dtype=torch.float32).reshape(16, 4),
        "region_ids": torch.tensor([0] * 10 + [1] * 6),
        "pair_indices": torch.arange(100, 116),
    }

    result = retriever.retrieve_q(query, bank, chunk_size=1)

    assert result["parent_keys"].shape == (1, 2, 4, 4)
    assert result["paired_p2_keys"].shape == (1, 2, 4, 4)
    assert result["valid"].all()
    assert result["query_valid"].tolist() == [True]
    assert bool((result["indices"][0, 0] < 110).all())
    assert bool((result["indices"][0, 1] >= 110).all())


def test_balanced_parent_retrieval_padding_and_missing_side_are_exact() -> None:
    retriever = BalancedParentRetriever(dim=4, topk_per_region=4)
    query = torch.randn(2, 4)
    bank = {
        "p3_keys": torch.randn(2, 4),
        "p2_keys": torch.randn(2, 4),
        "region_ids": torch.zeros(2, dtype=torch.long),
        "pair_indices": torch.tensor([7, 9]),
    }

    result = retriever.retrieve_q(query, bank)

    assert result["valid"][:, 0].sum(dim=-1).tolist() == [2, 2]
    assert not result["valid"][:, 1].any()
    assert not result["query_valid"].any()
    assert torch.count_nonzero(result["parent_keys"][:, 1]) == 0
    assert torch.count_nonzero(result["paired_p2_keys"][:, 1]) == 0
    assert bool((result["scores"][:, 1] == -1.0e4).all())
    assert bool((result["indices"][:, 1] == -1).all())


def test_child_query_uses_aligned_three_by_three_p2_patches() -> None:
    torch.manual_seed(3)
    builder = ChildQueryBuilder(p2_ch=1, dim=8, window=3)
    p2 = torch.zeros(2, 1, 5, 5)
    p2[0] = 1.0
    p2[1] = 7.0

    batch_ids = torch.tensor([0, 1])
    flat_indices = torch.tensor([12, 12])
    result = builder(p2, batch_ids, flat_indices, p3_hw=(5, 5))

    patches = result["child_patches"]
    assert patches.shape == (2, 1, 3, 3)
    torch.testing.assert_close(
        patches[0],
        torch.ones_like(patches[0]),
    )
    torch.testing.assert_close(
        patches[1],
        torch.full_like(patches[1], 7.0),
    )
    assert result["q_child"].shape == (2, 8)
    assert torch.equal(result["flat_indices2_from_p3"], flat_indices)
    torch.testing.assert_close(
        result["q_child"].norm(dim=-1), torch.ones(2), atol=1.0e-5, rtol=0
    )
