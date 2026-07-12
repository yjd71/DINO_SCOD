"""Build labeled-only DINO PC-HBM entries from baseline decoder features."""

from __future__ import annotations

from typing import Dict, Iterable, Sequence

import torch
import torch.nn.functional as F

from .common.utils import gather_tokens
from .memory.pc_region_builder import build_pc_regions
from .memory.sampling_policy import RegionSamplingRule, sample_region_indices


class DinoMemoryBuilder:
    """Convert one deterministic labeled batch into appendable memory entries."""

    def __init__(self, cfg, router, parent_retriever, child_query) -> None:
        self.cfg = cfg
        self.router = router
        self.parent_retriever = parent_retriever
        self.child_query = child_query

    def _sampling_rules(self) -> Dict[str, RegionSamplingRule]:
        return {
            name: RegionSamplingRule(max_count=max_count, min_count=min_count, ratio=ratio)
            for name, max_count, min_count, ratio in zip(
                self.cfg.region_names,
                self.cfg.region_max_quota,
                self.cfg.region_min_quota,
                self.cfg.region_sampling_ratio,
            )
        }

    @torch.no_grad()
    def __call__(
        self,
        features: Dict[str, torch.Tensor],
        gt: torch.Tensor,
        image_ids: Sequence[str],
    ) -> Dict[str, Dict[str, object]]:
        required = {'x3', 'p3', 'p2', 'm3', 'm2'}
        missing = required.difference(features)
        if missing:
            raise KeyError(f'Memory features are missing keys: {sorted(missing)}')

        x3 = features['x3']
        p3 = features['p3']
        child_map = features['p2']
        if x3.dim() != 4 or p3.shape != x3.shape or child_map.shape != x3.shape:
            raise ValueError('x3, p3 and p2 memory maps must share [B,128,H,W].')
        batch_size, _, height, width = x3.shape
        if len(image_ids) != batch_size:
            raise ValueError(f'Expected {batch_size} image ids, got {len(image_ids)}.')

        prob3 = torch.sigmoid(features['m3'])
        route = self.router.encode_route_tokens(x3, prob3=prob3)
        regions = build_pc_regions(
            gt,
            target_size=(height, width),
            boundary_kernel=self.cfg.fg_boundary_kernel,
            bg_near_kernel=self.cfg.bg_near_kernel,
            reliability_scale=self.cfg.sdf_reliability_scale,
        )
        regions = {
            key: value.to(device=x3.device, dtype=x3.dtype if value.is_floating_point() else None)
            for key, value in regions.items()
        }

        rules = self._sampling_rules()
        batch_indices = []
        flat_indices = []
        region_ids = []
        parent_meta = []
        for batch_index, raw_image_id in enumerate(image_ids):
            image_id = str(raw_image_id)
            reliability = regions['geometry'][batch_index, 5]
            for region_id, region_name in enumerate(self.cfg.region_names):
                selected = sample_region_indices(
                    regions[region_name][batch_index, 0].bool(),
                    reliability,
                    region_name,
                    rules=rules,
                )
                for flat_index in selected.tolist():
                    row = int(flat_index) // width
                    col = int(flat_index) % width
                    batch_indices.append(batch_index)
                    flat_indices.append(int(flat_index))
                    region_ids.append(region_id)
                    parent_meta.append(
                        {
                            'image_id': image_id,
                            'region': region_name,
                            'region_id': region_id,
                            'flat_index': int(flat_index),
                            'coord': (row, col),
                            'reliability': float(reliability.flatten()[flat_index].item()),
                        }
                    )

        device = x3.device
        batch_ids = torch.as_tensor(batch_indices, device=device, dtype=torch.long)
        parent_flat = torch.as_tensor(flat_indices, device=device, dtype=torch.long)
        region_ids_tensor = torch.as_tensor(region_ids, device=device, dtype=torch.long)

        key_map = self.parent_retriever.encode_k_map(p3)
        if parent_flat.numel() == 0:
            parent_keys = x3.new_empty((0, self.cfg.memory_dim))
            child_keys = x3.new_empty((0, self.cfg.memory_dim))
            child_patches = x3.new_empty(
                (0, x3.shape[1], self.cfg.child_window_size, self.cfg.child_window_size)
            )
            parent_geometry = x3.new_empty((0, self.cfg.geometry_dim))
            parent_values = x3.new_empty((0, self.cfg.value_dim))
        else:
            parent_keys = gather_tokens(key_map, batch_ids, parent_flat)
            child_encoded = self.child_query.encode_child_map(
                child_map,
                batch_ids,
                parent_flat,
                p3_hw=(height, width),
            )
            child_keys = child_encoded['q_child']
            child_patches = child_encoded['child_patches']
            parent_geometry = gather_tokens(
                regions['geometry'], batch_ids, parent_flat
            )
            one_hot = F.one_hot(region_ids_tensor, num_classes=4).to(dtype=x3.dtype)
            is_fg = (region_ids_tensor < 2).to(dtype=x3.dtype).unsqueeze(-1)
            is_bg = 1.0 - is_fg
            sdf = parent_geometry[:, 0:1]
            reliability = parent_geometry[:, 5:6]
            parent_values = torch.cat(
                [one_hot, is_fg, is_bg, sdf, reliability], dim=-1
            )

        child_ptr = torch.arange(parent_keys.shape[0], device=device, dtype=torch.long)
        child_meta = [dict(meta) for meta in parent_meta]
        route_entries = {
            key: value for key, value in route.items()
            if key.startswith('x3_') or key in {'route_embed'}
        }
        route_entries['img_ids'] = [str(image_id) for image_id in image_ids]

        return {
            'route': route_entries,
            'parent': {
                'p3_keys': parent_keys,
                'p3_values': parent_values,
                'p3_geometry': parent_geometry,
                'child_ptr': child_ptr,
                'parent_meta': parent_meta,
            },
            'child': {
                'p2_child_keys': child_keys,
                'p2_child_geo': parent_geometry.clone(),
                'child_meta': child_meta,
                'child_patches': child_patches,
            },
        }
