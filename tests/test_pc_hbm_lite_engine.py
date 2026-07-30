from __future__ import annotations

from dataclasses import dataclass

import torch

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.PC_HBM.dino_engine import DinoPCHBMEngine


@dataclass(frozen=True)
class _Compatible:
    compatible: bool = True
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.compatible


class _PairMemory:
    def __init__(self, *, both_regions: bool = True, compatible: bool = True):
        torch.manual_seed(17)
        count = 8 if both_regions else 4
        self.pairs = {
            "p3_keys": torch.randn(count, 128),
            "p2_keys": torch.randn(count, 128),
            "region_ids": torch.tensor(
                [0, 0, 0, 0] + ([1, 1, 1, 1] if both_regions else [])
            ),
            "pair_indices": torch.arange(count),
        }
        self.compatible = compatible

    def is_ready(self) -> bool:
        return True

    def validate_compat(self, expected):
        return _Compatible(self.compatible, None if self.compatible else "schema")

    def route_query(
        self,
        *,
        q_global,
        q_environment,
        top_img_k,
        query_image_ids=None,
        exclude_self_match=True,
    ):
        batch = q_global.size(0)
        return {
            "top_img_ids": [["memory"] for _ in range(batch)],
            "scores": torch.zeros(batch, top_img_k),
            "valid": torch.ones(batch, top_img_k, dtype=torch.bool),
            "indices": torch.zeros(batch, top_img_k, dtype=torch.long),
        }

    def get_pair_subbank(
        self,
        top_img_ids,
        *,
        device=None,
        dtype=None,
        exclude_image_id=None,
    ):
        return {
            key: (
                value.to(
                    device=device,
                    dtype=dtype if value.is_floating_point() else value.dtype,
                )
                if isinstance(value, torch.Tensor)
                else value
            )
            for key, value in self.pairs.items()
        }


def _features(batch: int = 1):
    torch.manual_seed(19)
    return (
        torch.randn(batch, 128, 28, 28),
        torch.randn(batch, 128, 28, 28),
        torch.randn(batch, 128, 28, 28),
        torch.randn(batch, 1, 28, 28),
    )


def test_engine_verify_only_runs_pairs_but_is_identity() -> None:
    engine = DinoPCHBMEngine(DinoPCHBMConfig())
    x3, p3, p2, m3 = _features()
    result = engine.forward_lite(
        x3, p3, p2, m3, _PairMemory(), mode="verify_only"
    )

    assert torch.equal(result["p3_corr"], p3)
    assert torch.count_nonzero(result["p3_delta"]) == 0
    assert result["p3_delta"].shape == (64, 128)
    assert result["pair_logits"].shape == (64, 2)
    assert result["retrieval_valid"].shape == (64, 2, 4)
    assert result["parent_cosine"].shape == (64, 2, 4)
    assert result["child_cosine"].shape == (64, 2, 4)
    assert result["query_valid"].all()
    assert result["query_mask_map"].sum().item() == 64
    scalar_maps = {
        key
        for key, value in result.items()
        if torch.is_tensor(value)
        and value.ndim == 4
        and value.size(1) == 1
        and key.endswith("_map")
    }
    assert scalar_maps == {
        "query_mask_map",
        "memory_confidence_map",
        "gate_map",
    }
    assert not list(engine.query_selector.parameters())


def test_engine_invalid_side_has_zero_correction_confidence_and_gate() -> None:
    engine = DinoPCHBMEngine(DinoPCHBMConfig())
    x3, p3, p2, m3 = _features()
    result = engine.forward_lite(
        x3, p3, p2, m3, _PairMemory(both_regions=False), mode="full"
    )

    assert not result["query_valid"].any()
    assert result["p3_delta"].shape == (64, 128)
    assert torch.count_nonzero(result["p3_delta"]) == 0
    assert torch.count_nonzero(result["memory_confidence_map"]) == 0
    assert torch.count_nonzero(result["gate_map"]) == 0
    assert torch.equal(result["p3_corr"], p3)
    assert torch.isfinite(result["pair_logits"]).all()
