"""Build labeled PC-HBM-Lite route descriptors and aligned P3/P2 pairs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .memory.pc_region_builder import build_boundary_pair_regions
from .memory.sampling_policy import rules_from_config, sample_region_indices


class DinoMemoryBuilder:
    """Convert one deterministic labeled batch into appendable Schema V2 entries."""

    def __init__(self, cfg, router, parent_retriever, child_query) -> None:
        self.cfg = cfg
        self.router = router
        self.parent_retriever = parent_retriever
        self.child_query = child_query

    @torch.no_grad()
    def __call__(
        self,
        features: Mapping[str, torch.Tensor],
        gt: torch.Tensor,
        image_ids: Sequence[str],
    ) -> dict[str, Any]:
        required = {"x3", "p3", "p2", "m3"}
        missing = required.difference(features)
        if missing:
            raise KeyError(f"Memory features are missing keys: {sorted(missing)}")
        unexpected = set(features).difference(required)
        if unexpected:
            raise KeyError(
                "Memory features contain unsupported keys: "
                f"{sorted(unexpected)}"
            )

        x3 = features["x3"]
        p3 = features["p3"]
        child_map = features["p2"]
        m3 = features["m3"]
        if x3.ndim != 4 or x3.size(1) != int(self.cfg.memory_dim):
            raise ValueError(
                f"x3 must be [B,{self.cfg.memory_dim},H,W], got {tuple(x3.shape)}"
            )
        if p3.shape != x3.shape or child_map.shape != x3.shape:
            raise ValueError("x3, p3, and p2 must share [B,128,H,W]")
        if (
            m3.ndim != 4
            or m3.size(0) != x3.size(0)
            or m3.size(1) != 1
            or m3.shape[-2:] != x3.shape[-2:]
        ):
            raise ValueError("m3 must be [B,1,H,W] and align with x3")
        batch_size, _, height, width = x3.shape
        if (height, width) != (int(self.cfg.token_size), int(self.cfg.token_size)):
            raise ValueError(
                f"Memory token grid is fixed to {self.cfg.token_size}x{self.cfg.token_size}"
            )
        normalized_image_ids = [str(image_id) for image_id in image_ids]
        if len(normalized_image_ids) != batch_size:
            raise ValueError(f"Expected {batch_size} image IDs, got {len(image_ids)}")
        if any(not image_id for image_id in normalized_image_ids):
            raise ValueError("Memory image IDs must be non-empty")
        if len(set(normalized_image_ids)) != len(normalized_image_ids):
            raise ValueError("Memory batch image IDs must be unique")

        route = self.router.encode_route_tokens(x3, torch.sigmoid(m3.float()))
        regions = build_boundary_pair_regions(
            gt,
            target_size=(height, width),
            boundary_kernel=int(self.cfg.fg_boundary_kernel),
            bg_near_kernel=int(self.cfg.bg_near_kernel),
            threshold=float(self.cfg.gt_binary_threshold),
        )
        rules = rules_from_config(self.cfg)

        batch_indices: list[int] = []
        flat_indices: list[int] = []
        region_ids: list[int] = []
        pair_meta: list[dict[str, Any]] = []
        for batch_index, image_id in enumerate(normalized_image_ids):
            for region_id, region_name in enumerate(self.cfg.region_names):
                selected = sample_region_indices(
                    regions[region_name][batch_index, 0],
                    region_name,
                    rules=rules,
                )
                for flat_index in selected.tolist():
                    row, col = divmod(int(flat_index), width)
                    batch_indices.append(batch_index)
                    flat_indices.append(int(flat_index))
                    region_ids.append(region_id)
                    pair_meta.append(
                        {
                            "image_id": image_id,
                            "region": str(region_name),
                            "region_id": region_id,
                            "flat_index": int(flat_index),
                            "coord": (row, col),
                            "source": "labeled_only",
                            "is_labeled": True,
                        }
                    )

        device = x3.device
        batch_ids = torch.tensor(batch_indices, device=device, dtype=torch.long)
        selected_flat = torch.tensor(flat_indices, device=device, dtype=torch.long)
        selected_regions = torch.tensor(region_ids, device=device, dtype=torch.long)
        encoded_p3_map = self.parent_retriever.encode_k_map(p3)
        if encoded_p3_map.shape != p3.shape:
            raise ValueError(
                "Parent key encoder must preserve [B,128,H,W], "
                f"got {tuple(encoded_p3_map.shape)}"
            )

        if selected_flat.numel() == 0:
            sampled_p3_keys = x3.new_empty((0, int(self.cfg.memory_dim)))
            sampled_p2_keys = x3.new_empty((0, int(self.cfg.memory_dim)))
        else:
            sampled_p3_keys = _gather_tokens(
                encoded_p3_map,
                batch_ids,
                selected_flat,
            )
            encoded_child = self.child_query.encode_child_map(
                child_map,
                batch_ids,
                selected_flat,
                p3_hw=(height, width),
            )
            if "q_child" not in encoded_child:
                raise KeyError("Child encoder output must contain q_child")
            sampled_p2_keys = encoded_child["q_child"]

        if sampled_p3_keys.shape != sampled_p2_keys.shape:
            raise RuntimeError("P3 and P2 pair keys must have identical shapes")
        if (
            sampled_p3_keys.size(0) != selected_regions.numel()
            or sampled_p3_keys.size(0) != len(pair_meta)
        ):
            raise RuntimeError("Pair tensors and metadata are not one-to-one")
        return {
            "source": "labeled_only",
            "route": {
                "global_keys": route["route_global"],
                "environment_keys": route["route_environment"],
                "img_ids": normalized_image_ids,
            },
            "pairs": {
                "p3_keys": sampled_p3_keys,
                "p2_keys": sampled_p2_keys,
                "region_ids": selected_regions,
                "pair_meta": pair_meta,
            },
        }


def _gather_tokens(
    feature_map: torch.Tensor,
    batch_ids: torch.Tensor,
    flat_indices: torch.Tensor,
) -> torch.Tensor:
    if batch_ids.shape != flat_indices.shape:
        raise ValueError("batch_ids and flat_indices must have identical shapes")
    if feature_map.ndim != 4:
        raise ValueError("feature_map must be [B,C,H,W]")
    flattened = feature_map.flatten(2).transpose(1, 2)
    return flattened[batch_ids, flat_indices]


__all__ = ["DinoMemoryBuilder"]
