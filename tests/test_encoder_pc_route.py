from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

from Model.PC_HBM.encoder.encoder_memory import (
    EncoderPCMemory,
    build_encoder_memory_compat_meta,
)
from Model.PC_HBM.encoder.encoder_router import EncoderPCRouter


def _unit(*components: tuple[int, float]) -> torch.Tensor:
    value = torch.zeros(128)
    for index, weight in components:
        value[index] = weight
    return F.normalize(value, dim=0)


def _memory(image_ids: tuple[str, ...], route_keys: torch.Tensor) -> EncoderPCMemory:
    image_count = len(image_ids)
    parents_per_image = 2
    parent_count = image_count * parents_per_image
    image_index = torch.arange(parent_count, dtype=torch.long) // parents_per_image
    reliability = torch.linspace(0.55, 0.90, parent_count)
    values = torch.zeros(parent_count, 8)
    values[torch.arange(parent_count), torch.arange(parent_count) % 4] = 1.0
    values[:, 4] = (torch.arange(parent_count) % 2 == 0).float()
    values[:, 5] = 1.0 - values[:, 4]
    values[:, 6] = torch.linspace(-1.0, 1.0, parent_count)
    values[:, 7] = reliability
    memory = EncoderPCMemory()
    memory.append(
        {
            "source": "labeled_only",
            "route": {
                "route_keys": route_keys,
                "cls4_keys": route_keys.clone(),
                "f4_global_keys": route_keys.clone(),
                "f3_boundary_keys": route_keys.clone(),
                "image_ids": list(image_ids),
            },
            "parent": {
                "f3_parent_keys": torch.randn(parent_count, 128),
                "values": values,
                "geometry": torch.randn(parent_count, 6),
                "child_ptr": torch.arange(parent_count, dtype=torch.long),
                "image_index": image_index,
                "region_id": torch.arange(parent_count, dtype=torch.long) % 4,
                "flat_index": torch.arange(parent_count, dtype=torch.long),
                "reliability": reliability,
            },
            "child": {
                "f2_child_keys": torch.randn(parent_count, 128),
                "f1_detail_keys": torch.randn(parent_count, 128),
                "geometry": torch.randn(parent_count, 6),
                "image_index": image_index.clone(),
                "flat_index": torch.arange(parent_count, dtype=torch.long),
            },
        }
    )
    memory.finalize(
        compat_meta=build_encoder_memory_compat_meta(
            dino_weight_fingerprint="route-test-dino-sha256",
            producer_fingerprint="route-test-adapter",
            labeled_split_fingerprint="route-test-split",
        )
    )
    return memory


def test_route_key_encoder_is_shared_normalized_and_gt_free() -> None:
    torch.manual_seed(12)
    router = EncoderPCRouter().eval()
    cls4 = torch.randn(2, 128)
    e4 = torch.randn(2, 128, 4, 4)
    e3 = torch.randn(2, 128, 4, 4)
    coarse = torch.sigmoid(torch.randn(2, 1, 4, 4))
    boundary = torch.sigmoid(torch.randn(2, 1, 4, 4))

    memory_encoding = router.encode_route_key(cls4, e4, e3, coarse, boundary)
    online_encoding = router.encode_route_key(cls4, e4, e3, coarse, boundary)

    assert torch.equal(memory_encoding["route_key"], online_encoding["route_key"])
    assert set(memory_encoding) == {
        "route_key",
        "cls4_key",
        "f4_global_key",
        "f3_boundary_key",
        "f3_uncertainty_key",
        "f3_environment_key",
    }
    for value in memory_encoding.values():
        assert value.shape == (2, 128)
        torch.testing.assert_close(value.norm(dim=1), torch.ones(2))
    signature = inspect.signature(router.encode_route_key)
    assert all("gt" not in name and "mask" not in name for name in signature.parameters)


def test_route_probability_aligns_amp_dtype_without_detaching_gradient() -> None:
    probability = torch.rand(2, 1, 14, 14, dtype=torch.float32, requires_grad=True)

    aligned = EncoderPCRouter._prepare_probability(
        probability,
        batch_size=2,
        spatial_size=(28, 28),
        name="coarse_probability",
        dtype=torch.float16,
        device=torch.device("cpu"),
    )

    assert aligned.dtype == torch.float16
    assert aligned.shape == (2, 1, 28, 28)
    aligned.float().mean().backward()
    assert probability.grad is not None
    assert torch.isfinite(probability.grad).all()


def test_unmasked_same_image_infonce_and_masked_retrieval_are_separate() -> None:
    keys = torch.stack(
        (
            _unit((0, 1.0)),
            _unit((0, 0.9), (1, 0.4358899)),
            _unit((2, 1.0)),
            _unit((2, 0.9), (3, 0.4358899)),
        )
    )
    memory = _memory(("A", "B", "C", "D"), keys)
    router = EncoderPCRouter(top_img_k=1)
    queries = torch.stack((_unit((0, 1.0)), _unit((2, 1.0))))

    result = router.route(
        queries,
        memory,
        query_image_ids=("A", "C"),
        require_same_image_positive=True,
    )

    assert result["positive_memory_image_index"].tolist() == [0, 2]
    torch.testing.assert_close(
        result["route_info_nce"],
        F.cross_entropy(
            result["route_logits"], result["positive_memory_image_index"]
        ),
    )
    torch.testing.assert_close(
        result["route_logits"][0, 0], torch.tensor(1.0 / 0.07)
    )
    # The labeled positive remains present in InfoNCE logits, but is excluded
    # before actual top-k retrieval.
    assert result["top_img_indices"][:, 0].tolist() == [1, 3]
    assert result["top_img_ids"] == [["B"], ["D"]]


def test_each_batch_item_gets_its_own_parent_subbank() -> None:
    keys = torch.stack(
        (
            _unit((0, 1.0)),
            _unit((0, 0.9), (1, 0.4358899)),
            _unit((2, 1.0)),
            _unit((2, 0.9), (3, 0.4358899)),
        )
    )
    memory = _memory(("A", "B", "C", "D"), keys)
    router = EncoderPCRouter(top_img_k=1)
    result = router.route(
        torch.stack((_unit((0, 1.0)), _unit((2, 1.0)))),
        memory,
        query_image_ids=("A", "C"),
    )

    first, second = result["parent_subbanks"]
    assert first["global_parent_indices"].tolist() == [2, 3]
    assert second["global_parent_indices"].tolist() == [6, 7]
    assert first["image_index"].unique().tolist() == [1]
    assert second["image_index"].unique().tolist() == [3]
    assert not torch.equal(
        result["routed_parent_indices"][0],
        result["routed_parent_indices"][1],
    )


def test_margin_confidence_has_floor_and_does_not_use_route_entropy() -> None:
    # Equal top-1/top-2 scores give maximum normalized entropy.  Confidence is
    # still the margin confidence with its floor, not 1-route_entropy.
    keys = torch.stack(
        (
            _unit((0, 1.0)),
            _unit((1, 1.0)),
            _unit((1, 1.0)),
        )
    )
    memory = _memory(("A", "B", "C"), keys)
    result = EncoderPCRouter(top_img_k=2).route(
        _unit((1, 1.0)).unsqueeze(0),
        memory,
        query_image_ids=("A",),
    )

    assert result["route_valid"].item()
    torch.testing.assert_close(result["route_margin"], torch.zeros(1))
    torch.testing.assert_close(result["route_entropy_norm"], torch.ones(1))
    torch.testing.assert_close(
        result["route_confidence"], torch.tensor([0.50]), rtol=0.0, atol=1.0e-6
    )


def test_margin_confidence_uses_raw_cosine_not_temperature_scaled_score() -> None:
    query = _unit((0, 1.0))
    top = _unit((0, 0.80), (1, 0.60))
    second = _unit((0, 0.76), (1, 0.6499231))
    memory = _memory(
        ("self", "top", "second"),
        torch.stack((query, top, second)),
    )
    router = EncoderPCRouter(
        top_img_k=2,
        tau_route=0.07,
        margin_temperature=0.03,
    )
    result = router.route(
        query.unsqueeze(0), memory, query_image_ids=("self",)
    )

    cosine_margin = result["top_img_similarities"][0, 0] - result[
        "top_img_similarities"
    ][0, 1]
    expected = torch.sigmoid(cosine_margin / 0.03).clamp_min(0.20)
    scaled_score_confidence = torch.sigmoid(
        (result["top_img_scores"][0, 0] - result["top_img_scores"][0, 1])
        / 0.03
    )
    torch.testing.assert_close(result["route_margin"][0], cosine_margin)
    torch.testing.assert_close(result["route_confidence"][0], expected)
    assert result["route_margin"].item() == pytest.approx(0.04, abs=5.0e-4)
    assert result["route_confidence"].item() == pytest.approx(
        torch.sigmoid(torch.tensor(0.04 / 0.03)).item(), abs=2.0e-3
    )
    assert not torch.isclose(result["route_confidence"][0], scaled_score_confidence)


def test_fewer_than_two_candidates_is_invalid_at_confidence_floor() -> None:
    memory = _memory(
        ("A", "B"),
        torch.stack((_unit((0, 1.0)), _unit((1, 1.0)))),
    )
    result = EncoderPCRouter(top_img_k=8).route(
        _unit((0, 1.0)).unsqueeze(0),
        memory,
        query_image_ids=("A",),
    )

    assert result["top_img_valid"].sum().item() == 1
    assert not result["route_valid"].item()
    assert result["route_margin_confidence"].item() == 0.0
    assert result["route_confidence"].item() == pytest.approx(0.20)


def test_labeled_route_fails_when_same_image_positive_is_missing() -> None:
    memory = _memory(
        ("A", "B"),
        torch.stack((_unit((0, 1.0)), _unit((1, 1.0)))),
    )
    router = EncoderPCRouter()
    with pytest.raises(RuntimeError, match="positive is missing"):
        router.route(
            _unit((0, 1.0)).unsqueeze(0),
            memory,
            query_image_ids=("not-in-memory",),
            require_same_image_positive=True,
        )

    unlabeled = router.route(_unit((0, 1.0)).unsqueeze(0), memory)
    assert unlabeled["route_info_nce"] is None
    assert unlabeled["positive_memory_image_index"].tolist() == [-1]


def test_route_rejects_logits_in_probability_inputs() -> None:
    router = EncoderPCRouter()
    cls4 = torch.randn(1, 128)
    e4 = torch.randn(1, 128, 4, 4)
    e3 = torch.randn(1, 128, 4, 4)
    with pytest.raises(ValueError, match="probabilities"):
        router.encode_route_key(
            cls4,
            e4,
            e3,
            torch.full((1, 1, 4, 4), 2.0),
            torch.full((1, 1, 4, 4), 0.5),
        )
