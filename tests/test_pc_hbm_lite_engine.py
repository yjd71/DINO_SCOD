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
    def __init__(
        self,
        *,
        both_regions: bool = True,
        compatible: bool = True,
        dim: int = 128,
    ):
        torch.manual_seed(17)
        count = 8 if both_regions else 4
        self.pairs = {
            "p3_keys": torch.randn(count, dim),
            "p2_keys": torch.randn(count, dim),
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
        global_weight=None,
        environment_weight=None,
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


def test_engine_uses_non_default_runtime_hyperparameters() -> None:
    cfg = DinoPCHBMConfig(
        input_size=140,
        token_size=10,
        output_size=35,
        decoder_dim=64,
        memory_dim=64,
        dino_layer_indices=(10, 1, 7, 4),
        route_top_img_k=3,
        route_global_weight=0.7,
        route_environment_weight=0.3,
        route_environment_min_mass=0.0,
        p3_top_ratio=0.2,
        p3_min_tokens=0,
        p3_max_tokens=100,
        query_boundary_weight=0.8,
        query_uncertainty_weight=0.2,
        fg_boundary_kernel=5,
        bg_near_kernel=9,
        region_names=("edge", "near"),
        region_max_quota=(20, 24),
        region_min_quota=(2, 3),
        region_sampling_ratio=(0.25, 0.75),
        parent_topk_per_region=2,
        query_chunk_size=7,
        child_window_size=5,
        tau_parent=0.2,
        tau_child=0.3,
    )
    engine = DinoPCHBMEngine(cfg)
    x3 = torch.randn(1, 64, 10, 10)
    p3 = torch.randn_like(x3)
    p2 = torch.randn_like(x3)
    m3 = torch.randn(1, 1, 10, 10)

    result = engine.forward_lite(
        x3,
        p3,
        p2,
        m3,
        _PairMemory(dim=64),
        mode="full",
    )

    assert result["query_flat_indices"].numel() == 20
    assert result["retrieval_valid"].shape == (20, 2, 2)
    assert result["pair_logits"].shape == (20, 2)
    assert result["p3_corr"].shape == p3.shape
    assert torch.isfinite(result["p3_corr"]).all()
    assert engine.child_query.window == 5
    assert engine.router.top_img_k == 3
