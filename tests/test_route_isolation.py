from __future__ import annotations

import torch

from Model.PC_HBM.common import gather_local_patches, merge_parent_results
from Model.PC_HBM.memory import PCMemory
from Model.PC_HBM.retrieval import ChildVerifierV2, ParentRetriever


def _routed_memory() -> PCMemory:
    dim = 128
    image_ids = ["A1", "A2", "B1", "B2"]
    route_embed = torch.zeros(4, dim)
    route_embed[0:2, 0] = 1.0
    route_embed[2:4, 1] = 1.0
    route = {
        name: route_embed.clone()
        for name in (
            "x3_global",
            "x3_boundary",
            "x3_uncertain",
            "x3_bg_near",
            "x3_environment",
            "route_embed",
        )
    }
    route["img_ids"] = image_ids
    parent_keys = torch.zeros(8, dim)
    parent_meta = []
    child_meta = []
    for image_index, image_id in enumerate(image_ids):
        parent_keys[2 * image_index : 2 * image_index + 2, image_index] = 1.0
        parent_meta.extend(
            [
                {"image_id": image_id, "region": "fg_boundary"},
                {"image_id": image_id, "region": "bg_near"},
            ]
        )
        child_meta.extend(
            [
                {"image_id": image_id, "region": "fg_boundary"},
                {"image_id": image_id, "region": "bg_near"},
            ]
        )
    values = torch.zeros(8, 8)
    values[:, 0] = 1.0
    values[:, 4] = 1.0
    values[:, 7] = 1.0
    memory = PCMemory()
    memory.append(
        {
            "source": "labeled_only",
            "route": route,
            "parent": {
                "p3_keys": parent_keys,
                "p3_values": values,
                "p3_geometry": torch.zeros(8, 6),
                "child_ptr": torch.arange(8),
                "parent_meta": parent_meta,
            },
            "child": {
                "p2_child_keys": parent_keys.clone(),
                "p2_child_geo": torch.zeros(8, 6),
                "child_meta": child_meta,
            },
        }
    )
    memory.finalize()
    return memory


def test_each_query_routes_and_retrieves_from_its_own_subbank() -> None:
    memory = _routed_memory()
    queries = torch.zeros(2, 128)
    queries[0, 0] = 1.0
    queries[1, 1] = 1.0
    route = memory.route_query(queries, top_img_k=2)
    assert set(route["top_img_ids"][0]) == {"A1", "A2"}
    assert set(route["top_img_ids"][1]) == {"B1", "B2"}

    retriever = ParentRetriever(128, dim=128, topk=4)
    all_results = []
    for batch_index in range(2):
        bank = memory.get_parent_subbank(route["top_img_ids"][batch_index], dtype=queries.dtype)
        result = retriever.retrieve_q(queries[batch_index : batch_index + 1], bank, chunk_size=1)
        result["output_positions"] = torch.tensor([batch_index])
        valid_meta = [
            item
            for item, valid in zip(result["top_parent_meta"][0], result["top_parent_valid"][0])
            if bool(valid)
        ]
        expected_prefix = "A" if batch_index == 0 else "B"
        assert all(item["image_id"].startswith(expected_prefix) for item in valid_meta)
        all_results.append(result)
    merged = merge_parent_results(all_results, total_queries=2)
    assert merged["top_parent_valid"].shape == (2, 4)
    assert merged["top_parent_valid"].all()


def test_self_match_exclusion_and_empty_route_do_not_fall_back_to_full_bank() -> None:
    memory = _routed_memory()
    query = torch.zeros(1, 128)
    query[:, 0] = 1.0
    route = memory.route_query(query, 1, query_image_ids=["A1"])
    assert route["top_img_ids"] == [["A2"]]
    assert memory.get_parent_subbank([])["p3_keys"].shape == (0, 128)
    assert memory.get_parent_subbank(None)["p3_keys"].shape == (0, 128)


def test_parent_candidate_padding_uses_invalid_mask_minus_one_and_negative_scores() -> None:
    memory = _routed_memory()
    bank = memory.get_parent_subbank(["A1"], dtype=torch.float32)
    retriever = ParentRetriever(128, dim=128, topk=4)
    query = torch.zeros(3, 128)
    query[:, 0] = 1.0
    result = retriever.retrieve_q(query, bank, chunk_size=2)
    assert result["top_parent_valid"].sum(dim=1).tolist() == [2, 2, 2]
    assert torch.all(result["top_child_ptrs"][:, 2:] == -1)
    assert torch.all(result["top_parent_indices"][:, 2:] == -1)
    assert torch.all(result["top_parent_scores"][:, 2:] == -1.0e4)
    assert torch.all(result["A_parent"][:, 2:] == 0)
    torch.testing.assert_close(result["A_parent"].sum(dim=1), torch.ones(3))


def test_child_patch_gather_is_batch_isolated() -> None:
    feature = torch.zeros(2, 1, 5, 5)
    feature[0] = 1.0
    feature[1] = 7.0
    patches = gather_local_patches(
        feature,
        batch_ids=torch.tensor([0, 1]),
        flat_indices=torch.tensor([12, 12]),
        window=5,
    )
    torch.testing.assert_close(patches[0], torch.ones_like(patches[0]))
    torch.testing.assert_close(patches[1], torch.full_like(patches[1], 7.0))


def test_all_invalid_child_hypotheses_produce_zero_attention_and_contradiction() -> None:
    query_count, topk, dim = 2, 4, 128
    verifier = ChildVerifierV2(dim=dim)
    parent = {
        "top_parent_keys": torch.zeros(query_count, topk, dim),
        "top_parent_values": torch.zeros(query_count, topk, 8),
        "top_parent_geo": torch.zeros(query_count, topk, 6),
        "top_parent_scores": torch.full((query_count, topk), -1.0e4),
        "top_parent_valid": torch.zeros(query_count, topk, dtype=torch.bool),
        "top_child_ptrs": torch.full((query_count, topk), -1, dtype=torch.long),
        "P3_group": torch.zeros(query_count, 4),
    }
    child = {
        "p2_child_keys": torch.zeros(query_count, topk, dim),
        "p2_child_geo": torch.zeros(query_count, topk, 6),
        "child_valid": torch.zeros(query_count, topk, dtype=torch.bool),
    }
    result = verifier(torch.zeros(query_count, dim), torch.zeros(query_count, 6), parent, child)
    assert not result["query_valid"].any()
    assert torch.count_nonzero(result["hyp_attn"]) == 0
    assert torch.count_nonzero(result["P_pc_group"]) == 0
    assert torch.count_nonzero(result["C23_token"]) == 0

