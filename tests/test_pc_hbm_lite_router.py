from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.PC_HBM.memory import PCMemory
from Model.PC_HBM.routing import CamouflageContextRouter


def _unit(index: int, *, sign: float = 1.0) -> torch.Tensor:
    value = torch.zeros(128)
    value[index] = sign
    return value


def _ready_route_memory() -> PCMemory:
    cfg = DinoPCHBMConfig()
    memory = PCMemory(config=cfg)
    global_keys = torch.stack((_unit(0), _unit(2), _unit(0), _unit(0, sign=-1)))
    environment_keys = torch.stack(
        (_unit(1, sign=-1), _unit(1), _unit(1), _unit(1, sign=-1))
    )
    image_ids = ["A", "B", "C", "D"]
    memory.append_route(
        global_keys=global_keys,
        environment_keys=environment_keys,
        img_ids=image_ids,
    )
    region_ids = torch.tensor([0, 1] * len(image_ids))
    pair_meta = [
        {"image_id": image_id, "region_id": region_id}
        for image_id in image_ids
        for region_id in (0, 1)
    ]
    pair_count = len(pair_meta)
    memory.append_pairs(
        p3_keys=F.normalize(torch.randn(pair_count, 128), dim=-1),
        p2_keys=F.normalize(torch.randn(pair_count, 128), dim=-1),
        region_ids=region_ids,
        pair_meta=pair_meta,
    )
    memory.finalize(compat_meta=cfg.expected_memory_meta())
    return memory


def test_router_is_parameter_free_and_returns_normalized_contexts() -> None:
    router = CamouflageContextRouter()
    assert sum(parameter.numel() for parameter in router.parameters()) == 0
    x3 = torch.randn(3, 128, 28, 28)
    prob3 = torch.rand(3, 1, 28, 28)
    output = router.encode_route_tokens(x3, prob3)
    assert set(output) == {"route_global", "route_environment"}
    assert output["route_global"].shape == (3, 128)
    assert output["route_environment"].shape == (3, 128)
    torch.testing.assert_close(output["route_global"].norm(dim=-1), torch.ones(3))
    torch.testing.assert_close(output["route_environment"].norm(dim=-1), torch.ones(3))


@pytest.mark.parametrize("probability", [0.0, 1.0])
def test_environment_fallback_is_finite_for_degenerate_predictions(
    probability: float,
) -> None:
    router = CamouflageContextRouter()
    x3 = torch.randn(2, 128, 28, 28, dtype=torch.float16)
    prob3 = torch.full((2, 1, 28, 28), probability, dtype=torch.float16)
    output = router.encode_route_tokens(x3, prob3)
    assert output["route_environment"].dtype == torch.float32
    assert torch.isfinite(output["route_environment"]).all()
    torch.testing.assert_close(
        output["route_environment"],
        output["route_global"],
        rtol=1.0e-5,
        atol=1.0e-5,
    )


def test_router_handles_empty_batch_without_nan() -> None:
    output = CamouflageContextRouter().encode_route_tokens(
        torch.empty(0, 128, 28, 28, dtype=torch.float16),
        torch.empty(0, 1, 28, 28, dtype=torch.float16),
    )
    assert output["route_global"].shape == (0, 128)
    assert output["route_environment"].shape == (0, 128)
    assert not torch.isnan(output["route_global"]).any()
    assert not torch.isnan(output["route_environment"]).any()


def test_weighted_route_order_and_self_match_exclusion() -> None:
    memory = _ready_route_memory()
    router = CamouflageContextRouter()
    query = torch.zeros(1, 128, 1, 1)
    query[:, 0] = 1.0
    probability = torch.zeros(1, 1, 1, 1)

    # Force the environment descriptor to e1 while retaining global descriptor e0.
    route = memory.route_query(
        q_global=_unit(0).unsqueeze(0),
        q_environment=_unit(1).unsqueeze(0),
        top_img_k=4,
    )
    assert route["top_img_ids"] == [["C", "B", "A", "D"]]
    torch.testing.assert_close(
        route["top_img_scores"][0],
        torch.tensor([1.0, 0.5, 0.0, -1.0]),
    )
    assert route["top_img_valid"].all()

    excluded = memory.route_query(
        q_global=_unit(0).unsqueeze(0),
        q_environment=_unit(1).unsqueeze(0),
        top_img_k=4,
        query_image_ids=["C"],
        exclude_self_match=True,
    )
    assert excluded["top_img_ids"] == [["B", "A", "D"]]
    assert excluded["top_img_valid"].tolist() == [[True, True, True, False]]
    assert excluded["top_img_indices"][0, -1].item() == -1
    assert torch.isfinite(excluded["route_entropy_norm"]).all()

    forwarded = router(
        query.expand(-1, -1, 28, 28),
        probability.expand(-1, -1, 28, 28),
        memory,
    )
    assert forwarded["top_img_scores"].shape == (1, 4)
    assert forwarded["route_global"].shape == (1, 128)


def test_router_non_default_and_extreme_weights_are_effective_and_finite() -> None:
    memory = _ready_route_memory()
    q_global = _unit(0).unsqueeze(0)
    q_environment = _unit(1).unsqueeze(0)

    global_heavy = memory.route_query(
        q_global,
        q_environment,
        4,
        global_weight=0.7,
        environment_weight=0.3,
    )
    environment_heavy = memory.route_query(
        q_global,
        q_environment,
        4,
        global_weight=0.3,
        environment_weight=0.7,
    )
    assert global_heavy["top_img_ids"] == [["C", "A", "B", "D"]]
    assert environment_heavy["top_img_ids"] == [["C", "B", "A", "D"]]

    extreme = memory.route_query(
        q_global,
        q_environment,
        4,
        global_weight=1.0e308,
        environment_weight=1.0e308,
    )
    assert torch.isfinite(extreme["top_img_scores"]).all()
    assert torch.isfinite(extreme["route_entropy_norm"]).all()

    class CaptureMemory:
        def __init__(self) -> None:
            self.kwargs = {}

        def route_query(self, **kwargs):
            self.kwargs = kwargs
            return {}

    capture = CaptureMemory()
    router = CamouflageContextRouter(
        global_weight=0.7,
        environment_weight=0.3,
    )
    router(
        torch.randn(1, 128, 2, 2),
        torch.rand(1, 1, 2, 2),
        capture,
    )
    assert capture.kwargs["global_weight"] == pytest.approx(0.7)
    assert capture.kwargs["environment_weight"] == pytest.approx(0.3)
