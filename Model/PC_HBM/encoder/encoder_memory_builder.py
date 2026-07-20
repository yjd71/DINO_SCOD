from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from ..memory.pc_region_builder import build_pc_regions
from ..memory.sampling_policy import rules_from_config, sample_region_indices


REGION_NAMES = ("fg_core", "fg_boundary", "bg_near", "bg_far")
MEMORY_FEATURE_FIELDS = (
    "route_keys",
    "cls4_keys",
    "f4_global_keys",
    "f3_boundary_keys",
    "f3_parent_keys",
    "f2_child_keys",
    "f1_detail_keys",
)


class EncoderMemoryBuilder:
    """Convert raw-bundle adapter features and labeled GT into schema-v3 entries.

    The Adapter owns every learned key projection and route computation. This
    builder uses GT only to form regions, geometry, sampling coordinates, and
    the fixed eight-channel prototype values.
    """

    def __init__(self, config: Any | None = None) -> None:
        self.config = config
        configured_names = tuple(getattr(config, "region_names", REGION_NAMES))
        if configured_names != REGION_NAMES:
            raise ValueError(
                "Encoder PC-HBM region_names must be "
                f"{REGION_NAMES}, got {configured_names}"
            )
        self.rules = rules_from_config(config)
        if tuple(self.rules) != REGION_NAMES:
            raise ValueError("Encoder PC-HBM sampling rules must cover the four fixed regions")
        self.memory_dim = int(getattr(config, "memory_dim", 128))
        self.value_dim = int(getattr(config, "value_dim", 8))
        self.geometry_dim = int(getattr(config, "geometry_dim", 6))
        self.token_size = int(getattr(config, "token_size", 28))
        if (self.memory_dim, self.value_dim, self.geometry_dim, self.token_size) != (
            128,
            8,
            6,
            28,
        ):
            raise ValueError("Encoder memory builder requires dimensions 128/8/6 and 28x28 tokens")

    def __call__(
        self,
        *,
        features: Mapping[str, torch.Tensor],
        gt: torch.Tensor,
        image_ids: Sequence[str],
    ) -> dict[str, Any]:
        if not isinstance(features, Mapping):
            raise TypeError("forward_memory_features must return a mapping")
        missing = [name for name in MEMORY_FEATURE_FIELDS if name not in features]
        if missing:
            raise KeyError(f"Encoder memory features are missing key: {missing[0]}")

        route = {
            name: _route_matrix(features[name], name, self.memory_dim)
            for name in (
                "route_keys",
                "cls4_keys",
                "f4_global_keys",
                "f3_boundary_keys",
            )
        }
        batch_size = _common_batch_size(route, "route")
        canonical_ids = [str(value).strip() for value in image_ids]
        if len(canonical_ids) != batch_size:
            raise ValueError(f"Expected {batch_size} image ids, got {len(canonical_ids)}")
        if any(not value for value in canonical_ids):
            raise ValueError("Encoder memory image ids must be non-empty")
        if len(set(canonical_ids)) != len(canonical_ids):
            raise ValueError("Encoder memory image ids must be unique within each batch")

        token_count = self.token_size * self.token_size
        f3_parent = _token_matrix(
            features["f3_parent_keys"],
            "f3_parent_keys",
            batch_size,
            token_count,
            self.memory_dim,
        )
        f2_child = _token_matrix(
            features["f2_child_keys"],
            "f2_child_keys",
            batch_size,
            token_count,
            self.memory_dim,
        )
        f1_detail = _token_matrix(
            features["f1_detail_keys"],
            "f1_detail_keys",
            batch_size,
            token_count,
            self.memory_dim,
        )
        feature_device = f3_parent.device
        feature_dtype = f3_parent.dtype
        for name, value in (*route.items(), ("f2_child_keys", f2_child), ("f1_detail_keys", f1_detail)):
            if value.device != feature_device:
                raise ValueError(f"All encoder memory features must share one device; {name} differs")
            if value.dtype != feature_dtype:
                raise ValueError(f"All encoder memory features must share one dtype; {name} differs")

        if not isinstance(gt, torch.Tensor):
            raise TypeError("Encoder memory GT must be a tensor")
        if int(gt.shape[0]) != batch_size:
            raise ValueError("Encoder memory features and GT batch dimensions differ")
        regions = build_pc_regions(
            gt,
            target_size=(self.token_size, self.token_size),
            boundary_kernel=int(getattr(self.config, "fg_boundary_kernel", 3)),
            bg_near_kernel=int(getattr(self.config, "bg_near_kernel", 7)),
            threshold=float(getattr(self.config, "memory_gt_threshold", 0.5)),
            reliability_scale=float(getattr(self.config, "sdf_reliability_scale", 0.15)),
        )
        geometry_map = regions["geometry"].to(
            device=feature_device, dtype=feature_dtype, non_blocking=True
        )

        batch_indices: list[torch.Tensor] = []
        flat_indices: list[torch.Tensor] = []
        region_ids: list[torch.Tensor] = []
        for batch_index in range(batch_size):
            reliability_map = regions["geometry"][batch_index, 5]
            for region_id, region_name in enumerate(REGION_NAMES):
                selected = sample_region_indices(
                    regions[region_name][batch_index, 0].bool(),
                    reliability_map,
                    region_name,
                    rules=self.rules,
                )
                if selected.numel() == 0:
                    continue
                selected = selected.to(device=feature_device, dtype=torch.long)
                flat_indices.append(selected)
                batch_indices.append(torch.full_like(selected, batch_index))
                region_ids.append(torch.full_like(selected, region_id))

        if not flat_indices:
            raise RuntimeError("No labeled encoder prototypes were sampled")
        flat_index = torch.cat(flat_indices, dim=0)
        image_index = torch.cat(batch_indices, dim=0)
        region_id = torch.cat(region_ids, dim=0)
        parent_keys = f3_parent[image_index, flat_index]
        child_keys = f2_child[image_index, flat_index]
        detail_keys = f1_detail[image_index, flat_index]
        geometry_tokens = _gather_map_tokens(geometry_map, image_index, flat_index)
        one_hot = F.one_hot(region_id, num_classes=4).to(dtype=feature_dtype)
        foreground = (region_id < 2).to(dtype=feature_dtype).unsqueeze(-1)
        background = 1.0 - foreground
        sdf = geometry_tokens[:, 0:1]
        reliability = geometry_tokens[:, 5]
        values = torch.cat(
            (one_hot, foreground, background, sdf, reliability.unsqueeze(-1)), dim=-1
        )
        if int(values.shape[1]) != self.value_dim:
            raise RuntimeError("Encoder memory value construction violated the 8-channel contract")

        child_ptr = torch.arange(parent_keys.shape[0], device=feature_device, dtype=torch.long)
        return {
            "source": "labeled_only",
            "route": {**route, "image_ids": canonical_ids},
            "parent": {
                "f3_parent_keys": parent_keys,
                "values": values,
                "geometry": geometry_tokens,
                "child_ptr": child_ptr,
                "image_index": image_index,
                "region_id": region_id,
                "flat_index": flat_index,
                "reliability": reliability,
            },
            "child": {
                "f2_child_keys": child_keys,
                "f1_detail_keys": detail_keys,
                "geometry": geometry_tokens.clone(),
                "image_index": image_index.clone(),
                "flat_index": flat_index.clone(),
            },
        }


def _route_matrix(value: object, name: str, width: int) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if value.ndim != 2 or int(value.shape[1]) != width:
        raise ValueError(f"{name} must have shape [B,{width}], got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _token_matrix(
    value: object,
    name: str,
    batch_size: int,
    token_count: int,
    width: int,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")
    if value.ndim == 4:
        if (
            int(value.shape[0]) != batch_size
            or int(value.shape[1]) != width
            or int(value.shape[2] * value.shape[3]) != token_count
        ):
            raise ValueError(
                f"{name} map must have shape [B,{width},28,28], got {tuple(value.shape)}"
            )
        value = value.flatten(2).transpose(1, 2).contiguous()
    if value.ndim != 3 or tuple(value.shape) != (batch_size, token_count, width):
        raise ValueError(
            f"{name} must have shape [B,{token_count},{width}], got {tuple(value.shape)}"
        )
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _common_batch_size(values: Mapping[str, torch.Tensor], group: str) -> int:
    counts = {int(value.shape[0]) for value in values.values()}
    if len(counts) != 1:
        raise ValueError(f"{group} features have inconsistent batch dimensions")
    return next(iter(counts), 0)


def _gather_map_tokens(
    feature_map: torch.Tensor,
    batch_index: torch.Tensor,
    flat_index: torch.Tensor,
) -> torch.Tensor:
    tokens = feature_map.flatten(2).transpose(1, 2)
    return tokens[batch_index, flat_index]


__all__ = ["EncoderMemoryBuilder", "MEMORY_FEATURE_FIELDS", "REGION_NAMES"]
