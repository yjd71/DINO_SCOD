from __future__ import annotations

import torch

from Model.PC_HBM.encoder.child_semantic_detail_verifier import (
    ChildSemanticDetailVerifier,
    EncoderParentChildDetailVerifier,
    EncoderParentRetriever,
    NormalizedStructuredPrior,
    _GeometrySupportScorer,
    build_support_targets,
)
from Model.PC_HBM.encoder.encoder_memory import (
    EncoderPCMemory,
    build_encoder_memory_compat_meta,
)
from configs.pc_hbm_dino_config import EncoderPCHBMConfig
from Model.PC_HBM.encoder.encoder_pc_adapter import EncoderPCHBMAdapter


def _memory(
    *,
    image_count: int = 2,
    parents_per_image: int = 20,
    invalid_child_ptrs: bool = False,
) -> EncoderPCMemory:
    torch.manual_seed(17)
    parent_count = image_count * parents_per_image
    image_index = torch.arange(image_count).repeat_interleave(parents_per_image)
    region_id = torch.arange(parent_count) % 4
    reliability = torch.linspace(0.55, 0.95, parent_count)
    values = torch.zeros(parent_count, 8)
    values[torch.arange(parent_count), region_id] = 1.0
    values[:, 4] = (region_id < 2).float()
    values[:, 5] = 1.0 - values[:, 4]
    values[:, 6] = torch.linspace(-0.9, 0.9, parent_count)
    values[:, 7] = reliability

    parent_keys = torch.randn(parent_count, 128)
    for image in range(image_count):
        parent_keys[image_index == image, image] += 8.0
    child_semantic = torch.randn(parent_count, 128)
    child_detail = torch.randn(parent_count, 128)
    geometry = torch.zeros(parent_count, 6)
    geometry[:, 0] = torch.linspace(-1.0, 1.0, parent_count)
    geometry[:, 1] = 1.0
    geometry[:, 3] = torch.linspace(-0.5, 0.5, parent_count)
    geometry[:, 5] = reliability
    child_ptr = torch.full((parent_count,), -1, dtype=torch.long)
    if not invalid_child_ptrs:
        child_ptr = torch.arange(parent_count)

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
                "f3_parent_keys": parent_keys,
                "values": values,
                "geometry": geometry,
                "child_ptr": child_ptr,
                "image_index": image_index,
                "region_id": region_id,
                "flat_index": torch.arange(parent_count) % 784,
                "reliability": reliability,
            },
            "child": {
                "f2_child_keys": child_semantic,
                "f1_detail_keys": child_detail,
                "geometry": geometry.clone(),
                "image_index": image_index.clone(),
                "flat_index": torch.arange(parent_count) % 784,
            },
        }
    )
    memory.finalize(
        compat_meta=build_encoder_memory_compat_meta(
            dino_weight_fingerprint="verify-dino-sha256",
            producer_fingerprint="verify-adapter-sha256",
            labeled_split_fingerprint="verify-split-sha256",
        )
    )
    return memory


def _subbank(memory: EncoderPCMemory, image_indices: tuple[int, ...]) -> dict[str, torch.Tensor]:
    selected = torch.nonzero(
        torch.isin(
            memory.parent["image_index"].long(),
            torch.tensor(image_indices, dtype=torch.long),
        ),
        as_tuple=False,
    ).flatten()
    result = {"global_parent_indices": selected}
    result.update(
        {name: value.index_select(0, selected) for name, value in memory.parent.items()}
    )
    return result


def _maps(batch_size: int = 2, height: int = 7, width: int = 7) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(23)
    return tuple(
        torch.randn(batch_size, 128, height, width, requires_grad=True)
        for _ in range(3)
    )


def _query_geometry(rows: int) -> torch.Tensor:
    geometry = torch.zeros(rows, 6)
    geometry[:, 1] = 1.0
    geometry[:, 5] = 0.8
    return geometry


def test_zero_geometry_reliability_has_exact_zero_and_finite_gradients() -> None:
    scorer = _GeometrySupportScorer()
    query = torch.tensor(
        [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True
    )
    parent = torch.tensor(
        [[[0.0, 1.0, 0.0, 0.0, 0.0, 0.8]]], requires_grad=True
    )
    child = torch.tensor(
        [[[0.0, 1.0, 0.0, 0.0, 0.0, 0.9]]], requires_grad=True
    )

    result = scorer(query, parent, child)

    assert torch.equal(result["geometry_reliability"], torch.zeros(1, 1))
    result["score"].sum().backward()
    assert query.grad is not None and torch.isfinite(query.grad).all()
    assert parent.grad is not None and torch.isfinite(parent.grad).all()
    assert child.grad is not None and torch.isfinite(child.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in scorer.parameters()
    )


def test_parent_retrieval_is_per_sample_top16_and_chunked_at_512() -> None:
    memory = _memory()
    retriever = EncoderParentRetriever()
    subbanks = [_subbank(memory, (0,)), _subbank(memory, (1,))]
    q3 = torch.randn(513, 128)
    batch_ids = torch.cat(
        (torch.zeros(257, dtype=torch.long), torch.ones(256, dtype=torch.long))
    )

    result = retriever.retrieve_q(q3, batch_ids, subbanks)

    assert retriever.topk == 16
    assert retriever.query_chunk_size == 512
    assert result["top_parent_keys"].shape == (513, 16, 128)
    assert result["top_parent_values"].shape == (513, 16, 8)
    assert result["top_parent_geometry"].shape == (513, 16, 6)
    assert torch.equal(result["top_parent_valid"].sum(dim=1), torch.full((513,), 16))
    assert torch.equal(
        result["top_parent_image_indices"][:257][result["top_parent_valid"][:257]],
        torch.zeros(257 * 16, dtype=torch.long),
    )
    assert torch.equal(
        result["top_parent_image_indices"][257:][result["top_parent_valid"][257:]],
        torch.ones(256 * 16, dtype=torch.long),
    )


def test_adapter_wires_smoke_parent_top2_and_preserves_default_top16() -> None:
    default_adapter = EncoderPCHBMAdapter()
    smoke_adapter = EncoderPCHBMAdapter(
        EncoderPCHBMConfig(parent_topk=2, query_chunk_size=32)
    )

    assert default_adapter.verifier.parent_retriever.topk == 16
    assert default_adapter.verifier.child_verifier.parent_topk == 16
    assert smoke_adapter.verifier.parent_retriever.topk == 2
    assert smoke_adapter.verifier.parent_retriever.query_chunk_size == 32
    assert smoke_adapter.verifier.child_verifier.parent_topk == 2


def test_parent_padding_uses_invalid_mask_without_full_bank_fallback() -> None:
    memory = _memory()
    subbank = _subbank(memory, (0,))
    small = {
        name: value[:3] if isinstance(value, torch.Tensor) else value
        for name, value in subbank.items()
    }
    retriever = EncoderParentRetriever()
    result = retriever.retrieve_q(
        torch.randn(1, 128), torch.zeros(1, dtype=torch.long), [small]
    )

    assert result["top_parent_valid"].sum().item() == 3
    assert torch.equal(result["top_parent_indices"][0, 3:], torch.full((13,), -1))
    assert torch.equal(result["top_parent_keys"][0, 3:], torch.zeros(13, 128))
    assert (result["top_parent_scores"][0, 3:] == -1.0e4).all()


def test_parent_retrieval_aligns_autocast_scores_to_query_dtype() -> None:
    memory = _memory()
    query = torch.randn(2, 128, dtype=torch.float32, requires_grad=True)
    retriever = EncoderParentRetriever()

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = retriever.retrieve_q(
            query,
            torch.zeros(2, dtype=torch.long),
            [_subbank(memory, (0,))],
        )

    assert result["top_parent_scores"].dtype == query.dtype
    result["top_parent_scores"].mean().backward()
    assert query.grad is not None
    assert torch.isfinite(query.grad).all()


def test_f2_f1_windows_shapes_geometry_and_vector_contract() -> None:
    memory = _memory()
    verifier = EncoderParentChildDetailVerifier()
    e1, e2, e3 = _maps()
    batch_ids = torch.tensor([0, 1])
    flat_indices = torch.tensor([24, 24])
    captured: dict[str, tuple[int, ...]] = {}
    handles = [
        verifier.child_verifier.semantic_local_encoder.register_forward_pre_hook(
            lambda _module, args: captured.__setitem__("semantic", tuple(args[0].shape))
        ),
        verifier.child_verifier.detail_local_encoder.register_forward_pre_hook(
            lambda _module, args: captured.__setitem__("detail", tuple(args[0].shape))
        ),
    ]
    try:
        result = verifier(
            e1,
            e2,
            e3,
            batch_ids,
            flat_indices,
            [_subbank(memory, (0,)), _subbank(memory, (1,))],
            memory,
            _query_geometry(2),
            query_region_ids=torch.tensor([1, 2]),
        )
    finally:
        for handle in handles:
            handle.remove()

    assert captured == {
        "semantic": (2, 128, 5, 5),
        "detail": (2, 128, 3, 3),
    }
    assert result["q_semantic"].shape == (2, 128)
    assert result["q_detail"].shape == (2, 128)
    assert result["top_child_semantic_keys"].shape == (2, 16, 128)
    assert result["top_child_detail_keys"].shape == (2, 16, 128)
    assert result["top_child_geometry"].shape == (2, 16, 6)
    assert result["S_semantic"].shape == (2, 16)
    assert result["S_detail"].shape == (2, 16)
    assert result["S_geometry"].shape == (2, 16)
    assert result["verified_evidence"].shape == (2, 128)
    assert result["query_valid"].all()
    assert all(reason == "" for reason in result["reason"])
    assert torch.isfinite(result["verified_evidence"]).all()

    loss = (
        result["verified_evidence"].square().mean()
        + result["semantic_support"].mean()
        + result["detail_support"].mean()
        + result["geometry_support"].mean()
    )
    loss.backward()
    assert e1.grad is not None and e1.grad.abs().sum() > 0
    assert e2.grad is not None and e2.grad.abs().sum() > 0
    assert e3.grad is not None and e3.grad.abs().sum() > 0


def test_structured_prior_is_normalized_and_residual_is_zero_gated() -> None:
    prior = NormalizedStructuredPrior()
    assert prior.gamma_prior.item() == 0.0
    parent = torch.tensor([[0.2, 0.7, -0.1]])
    semantic = torch.tensor([[0.4, -0.2, 0.8]])
    detail = torch.tensor([[0.1, 0.6, -0.3]])
    geometry = torch.tensor([[0.9, 0.3, -0.4]])
    valid = torch.tensor([[True, True, False]])
    before = prior(parent, semantic, detail, geometry, valid)
    with torch.no_grad():
        for parameter in prior.prior_residual_mlp.parameters():
            parameter.add_(100.0)
    after = prior(parent, semantic, detail, geometry, valid)

    assert torch.equal(before["prior_bias"], after["prior_bias"])
    expected_contradiction = (
        torch.abs(before["parent_normalized"] - before["semantic_normalized"])
        + torch.abs(before["semantic_normalized"] - before["detail_normalized"])
    ) / 2.0
    assert torch.allclose(before["contradiction"], expected_contradiction)
    for name in (
        "parent_normalized",
        "semantic_normalized",
        "detail_normalized",
        "geometry_normalized",
        "contradiction",
    ):
        assert ((before[name] >= 0.0) & (before[name] <= 1.0)).all()
        assert before[name][0, 2].item() == 0.0


def test_hard_negative_targets_are_labeled_only_and_do_not_change_evidence() -> None:
    candidates = torch.tensor([[1, 2, 0, 3], [2, 1, 3, 0]])
    valid = torch.ones_like(candidates, dtype=torch.bool)
    targets = build_support_targets(torch.tensor([1, 2]), candidates, valid)
    assert targets["semantic_hard_negative_mask"][0, 1]
    assert targets["semantic_hard_negative_mask"][1, 1]
    assert targets["detail_hard_negative_mask"][0, 2]
    assert targets["detail_hard_negative_mask"][1, 2]

    memory = _memory()
    verifier = EncoderParentChildDetailVerifier().eval()
    e1, e2, e3 = _maps()
    args = (
        e1,
        e2,
        e3,
        torch.tensor([0, 1]),
        torch.tensor([8, 8]),
        [_subbank(memory, (0,)), _subbank(memory, (1,))],
        memory,
        _query_geometry(2),
    )
    first = verifier(*args, query_region_ids=torch.tensor([0, 1]))
    second = verifier(*args, query_region_ids=torch.tensor([3, 2]))
    assert torch.equal(first["verified_evidence"], second["verified_evidence"])
    assert torch.equal(first["prior_bias"], second["prior_bias"])
    assert first["support_targets_available"].all()
    assert first["semantic_hard_negative_score"].shape == (2,)
    assert first["detail_hard_negative_score"].shape == (2,)


def test_empty_routed_subbank_returns_zero_evidence_and_reason() -> None:
    memory = _memory(image_count=1)
    verifier = EncoderParentChildDetailVerifier()
    e1, e2, e3 = _maps(batch_size=1)
    result = verifier(
        e1,
        e2,
        e3,
        torch.tensor([0]),
        torch.tensor([0]),
        [{}],
        memory,
        _query_geometry(1),
    )

    assert not result["parent_query_valid"].any()
    assert not result["query_valid"].any()
    assert result["reason"] == ("empty_routed_parent_subbank",)
    assert torch.equal(result["verified_evidence"], torch.zeros(1, 128))
    assert torch.equal(result["semantic_evidence"], torch.zeros(1, 128))
    assert torch.equal(result["hypothesis_attention"], torch.zeros(1, 16))
    assert torch.equal(result["top_parent_indices"], torch.full((1, 16), -1))


def test_all_invalid_children_return_zero_evidence_and_reason() -> None:
    memory = _memory(image_count=1, invalid_child_ptrs=True)
    verifier = EncoderParentChildDetailVerifier()
    e1, e2, e3 = _maps(batch_size=1)
    result = verifier(
        e1,
        e2,
        e3,
        torch.tensor([0]),
        torch.tensor([0]),
        [_subbank(memory, (0,))],
        memory,
        _query_geometry(1),
    )

    assert result["parent_query_valid"].all()
    assert not result["query_valid"].any()
    assert result["reason"] == ("all_child_hypotheses_invalid",)
    assert not result["child_valid"].any()
    assert torch.equal(result["verified_evidence"], torch.zeros(1, 128))
    assert torch.equal(result["prior_bias"], torch.zeros(1, 16))
    assert torch.equal(result["contradiction"], torch.zeros(1, 16))


def test_verifier_rejects_non_128_vectors_and_bad_shapes() -> None:
    for constructor in (EncoderParentRetriever, ChildSemanticDetailVerifier):
        try:
            constructor(dim=64)
        except ValueError as error:
            assert "128" in str(error)
        else:
            raise AssertionError("non-128 verifier construction must fail")
