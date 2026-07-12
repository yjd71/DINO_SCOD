"""CPU-FP16, labelled-only memory protocol for DINO PC-HBM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, Mapping, Sequence

import torch

from ..common.utils import EPS, REGION_TO_ID, entropy_from_probs, normalize


@dataclass(frozen=True)
class CompatibilityResult:
    """Boolean-compatible result that can also be unpacked as ``ok, reason``."""

    compatible: bool
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.compatible

    def __iter__(self) -> Iterator[object]:
        yield self.compatible
        yield self.reason


class PCMemory:
    """Route/parent/child tensor store with no persistent GPU cache.

    All stored floating tensors are detached CPU FP16 tensors.  Only a routed
    parent subbank (and its selected children) is transferred for a query.
    """

    FORMAT_VERSION = 1
    DEFAULT_SCHEMA_VERSION = 1

    def __init__(
        self,
        memory_dim: int = 128,
        value_dim: int = 8,
        geometry_dim: int = 6,
        *,
        storage_dtype: torch.dtype | str = torch.float16,
        compat_meta: Mapping[str, Any] | None = None,
        config: Any | None = None,
    ) -> None:
        self.memory_dim = int(memory_dim)
        self.value_dim = int(value_dim)
        self.geometry_dim = int(geometry_dim)
        self.storage_dtype = _parse_storage_dtype(storage_dtype)
        if self.storage_dtype != torch.float16:
            raise ValueError("PC-HBM memory storage dtype is fixed to float16")
        self.config = config
        self._initial_compat_meta = dict(compat_meta or {})
        self.clear()

    def clear(self) -> None:
        self._route_lists: dict[str, list[torch.Tensor]] = {
            "x3_global": [],
            "x3_boundary": [],
            "x3_uncertain": [],
            "x3_bg_near": [],
            "x3_environment": [],
            "route_embed": [],
        }
        self._route_img_ids: list[str] = []
        self._parent_key_list: list[torch.Tensor] = []
        self._parent_value_list: list[torch.Tensor] = []
        self._parent_geometry_list: list[torch.Tensor] = []
        self._parent_child_ptr_list: list[torch.Tensor] = []
        self._parent_meta_list: list[dict[str, Any]] = []
        self._child_key_list: list[torch.Tensor] = []
        self._child_geometry_list: list[torch.Tensor] = []
        self._child_meta_list: list[dict[str, Any]] = []
        self.route: dict[str, Any] = {}
        self.parent: dict[str, Any] = {}
        self.child: dict[str, Any] = {}
        self.compat_meta: dict[str, Any] = dict(self._initial_compat_meta)
        self.parent_img_to_indices: dict[str, torch.Tensor] = {}
        self.route_img_to_index: dict[str, int] = {}
        self._finalized = False

    def append(self, entries: Mapping[str, Any] | Sequence[Mapping[str, Any]]) -> None:
        """Append one builder output or a sequence of builder outputs."""

        if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes, Mapping)):
            for item in entries:
                self.append(item)
            return
        if not isinstance(entries, Mapping):
            raise TypeError("memory entries must be a mapping or sequence of mappings")
        source = str(entries.get("source", "labeled_only"))
        if source != "labeled_only":
            raise ValueError(f"PC-HBM accepts labeled_only entries, got {source!r}")
        if "compat_meta" in entries:
            self.compat_meta.update(dict(entries["compat_meta"] or {}))

        route = entries.get("route")
        if route:
            self.append_route(
                x3_global=route["x3_global"],
                x3_boundary=route["x3_boundary"],
                x3_uncertain=route["x3_uncertain"],
                x3_bg_near=route["x3_bg_near"],
                x3_environment=route["x3_environment"],
                route_embed=route["route_embed"],
                img_ids=route["img_ids"],
            )

        child = entries.get("child")
        child_offset = self.num_children_pending()
        if child:
            self.append_child(
                child["p2_child_keys"],
                child["p2_child_geo"],
                child.get("child_meta", [{} for _ in range(child["p2_child_keys"].size(0))]),
            )

        parent = entries.get("parent")
        if parent:
            pointers = parent["child_ptr"].detach().long().clone()
            if not bool(parent.get("child_ptr_is_global", False)):
                pointers = torch.where(pointers >= 0, pointers + child_offset, pointers)
            self.append_parent(
                parent["p3_keys"],
                parent["p3_values"],
                parent["p3_geometry"],
                pointers,
                parent.get("parent_meta", [{} for _ in range(parent["p3_keys"].size(0))]),
            )

    def append_route(
        self,
        *,
        x3_global: torch.Tensor,
        x3_boundary: torch.Tensor,
        x3_uncertain: torch.Tensor,
        x3_bg_near: torch.Tensor,
        x3_environment: torch.Tensor,
        route_embed: torch.Tensor,
        img_ids: Sequence[object],
    ) -> None:
        tensors = {
            "x3_global": x3_global,
            "x3_boundary": x3_boundary,
            "x3_uncertain": x3_uncertain,
            "x3_bg_near": x3_bg_near,
            "x3_environment": x3_environment,
            "route_embed": route_embed,
        }
        count: int | None = None
        for name, tensor in tensors.items():
            self._check_matrix(tensor, self.memory_dim, name)
            count = tensor.size(0) if count is None else count
            if tensor.size(0) != count:
                raise ValueError("All route descriptors must have the same image count")
        if count != len(img_ids):
            raise ValueError("Route descriptor count must match img_ids")
        normalized_ids = [str(image_id) for image_id in img_ids]
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("Duplicate image IDs within one route append are not allowed")
        existing = set(self._route_img_ids)
        duplicates = existing.intersection(normalized_ids)
        if duplicates:
            raise ValueError(f"Duplicate image IDs in labelled memory: {sorted(duplicates)}")
        for name, tensor in tensors.items():
            self._route_lists[name].append(self._store_float(tensor))
        self._route_img_ids.extend(normalized_ids)
        self._finalized = False

    def append_parent(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        geometry: torch.Tensor,
        child_ptr: torch.Tensor,
        meta: Sequence[Mapping[str, Any]],
    ) -> None:
        self._check_matrix(keys, self.memory_dim, "parent keys")
        self._check_matrix(values, self.value_dim, "parent values")
        self._check_matrix(geometry, self.geometry_dim, "parent geometry")
        count = keys.size(0)
        if values.size(0) != count or geometry.size(0) != count:
            raise ValueError("parent keys, values and geometry must have the same length")
        if child_ptr.numel() != count or len(meta) != count:
            raise ValueError("child_ptr and parent metadata must match parent count")
        normalized_meta = [_validate_labeled_meta(item, "parent") for item in meta]
        self._parent_key_list.append(self._store_float(keys))
        self._parent_value_list.append(self._store_float(values))
        self._parent_geometry_list.append(self._store_float(geometry))
        self._parent_child_ptr_list.append(child_ptr.detach().to(device="cpu", dtype=torch.long).view(-1))
        self._parent_meta_list.extend(normalized_meta)
        self._finalized = False

    def append_child(
        self,
        keys: torch.Tensor,
        geometry: torch.Tensor,
        meta: Sequence[Mapping[str, Any]],
    ) -> torch.Tensor:
        self._check_matrix(keys, self.memory_dim, "child keys")
        self._check_matrix(geometry, self.geometry_dim, "child geometry")
        count = keys.size(0)
        if geometry.size(0) != count or len(meta) != count:
            raise ValueError("child keys, geometry and metadata must have the same length")
        start = self.num_children_pending()
        self._child_key_list.append(self._store_float(keys))
        self._child_geometry_list.append(self._store_float(geometry))
        self._child_meta_list.extend(_validate_labeled_meta(item, "child") for item in meta)
        self._finalized = False
        return torch.arange(start, start + count, dtype=torch.long)

    def finalize(
        self,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float16,
        *,
        compat_meta: Mapping[str, Any] | None = None,
    ) -> None:
        device = torch.device(device)
        if device.type != "cpu" or dtype != torch.float16:
            raise ValueError("PC-HBM memory must be finalized as CPU float16")
        self.route = {
            name: _cat_or_empty(items, self.memory_dim, self.storage_dtype)
            for name, items in self._route_lists.items()
        }
        self.route["img_ids"] = list(self._route_img_ids)
        self.parent = {
            "p3_keys": _cat_or_empty(self._parent_key_list, self.memory_dim, self.storage_dtype),
            "p3_values": _cat_or_empty(self._parent_value_list, self.value_dim, self.storage_dtype),
            "p3_geometry": _cat_or_empty(self._parent_geometry_list, self.geometry_dim, self.storage_dtype),
            "child_ptr": _cat_long_or_empty(self._parent_child_ptr_list),
            "parent_meta": [dict(item) for item in self._parent_meta_list],
        }
        self.child = {
            "p2_child_keys": _cat_or_empty(self._child_key_list, self.memory_dim, self.storage_dtype),
            "p2_child_geo": _cat_or_empty(self._child_geometry_list, self.geometry_dim, self.storage_dtype),
            "child_meta": [dict(item) for item in self._child_meta_list],
        }
        if compat_meta is not None:
            self.compat_meta.update(dict(compat_meta))
        self.compat_meta.setdefault("schema_version", self.DEFAULT_SCHEMA_VERSION)
        self.compat_meta.setdefault("source", "labeled_only")
        self.compat_meta.setdefault("storage_dtype", "float16")
        self.compat_meta.setdefault("memory_dim", self.memory_dim)
        self.compat_meta.setdefault("value_dim", self.value_dim)
        self.compat_meta.setdefault("geometry_dim", self.geometry_dim)
        self._finalized = True
        self._build_indices()
        self._validate_finalized_storage()

    def is_ready(self) -> bool:
        if not self._finalized:
            return False
        return (
            self.route.get("route_embed", torch.empty(0, self.memory_dim)).size(0) > 0
            and self.parent.get("p3_keys", torch.empty(0, self.memory_dim)).size(0) > 0
            and self.child.get("p2_child_keys", torch.empty(0, self.memory_dim)).size(0) > 0
        )

    def validate_compat(
        self,
        expected: Mapping[str, Any] | object | None,
        *,
        require_producer_match: bool = False,
    ) -> CompatibilityResult:
        """Validate architecture/schema/dimensions and, optionally, producer."""

        if not self.is_ready():
            return CompatibilityResult(False, "memory_not_ready")
        if expected is None:
            return CompatibilityResult(True, None)
        if not isinstance(expected, Mapping):
            builder = getattr(expected, "expected_memory_meta", None)
            if callable(builder):
                expected = builder()
            else:
                raise TypeError("expected compatibility data must be a mapping or PC config")
        keys = (
            "architecture",
            "schema_version",
            "input_size",
            "token_hw",
            "output_hw",
            "dino_layer_indices",
            "encoder_dim",
            "decoder_dim",
            "memory_dim",
            "value_dim",
            "geometry_dim",
            "storage_dtype",
            "source",
        )
        if require_producer_match:
            keys = (*keys, "producer_fingerprint")
        for key in keys:
            if key not in expected:
                continue
            if key not in self.compat_meta:
                return CompatibilityResult(False, f"missing_compat_key:{key}")
            if _canonical_meta(self.compat_meta[key]) != _canonical_meta(expected[key]):
                return CompatibilityResult(False, f"compat_mismatch:{key}")
        return CompatibilityResult(True, None)

    def route_query(
        self,
        q_route: torch.Tensor,
        top_img_k: int,
        *,
        query_image_ids: Sequence[object] | None = None,
        exclude_self_match: bool = True,
    ) -> Dict[str, Any]:
        """Route each query independently and optionally exclude its own image."""

        if q_route.ndim != 2 or q_route.size(1) != self.memory_dim:
            raise ValueError(f"q_route must be [B,{self.memory_dim}], got {tuple(q_route.shape)}")
        batch_size = q_route.size(0)
        k = max(0, int(top_img_k))
        if query_image_ids is not None and len(query_image_ids) != batch_size:
            raise ValueError("query_image_ids length must match route query batch")
        output = self._empty_route_result(q_route, k)
        if not self.is_ready() or k == 0:
            return output

        keys = self.route["route_embed"].to(
            device=q_route.device,
            dtype=q_route.dtype,
            non_blocking=True,
        )
        similarities = normalize(q_route, dim=-1) @ normalize(keys, dim=-1).transpose(0, 1)
        image_ids = list(self.route["img_ids"])
        score_rows: list[torch.Tensor] = []
        valid_rows: list[torch.Tensor] = []
        index_rows: list[torch.Tensor] = []
        top_ids: list[list[str]] = []
        for batch_index in range(batch_size):
            valid_candidates = torch.ones(len(image_ids), device=q_route.device, dtype=torch.bool)
            if exclude_self_match and query_image_ids is not None:
                own_id = str(query_image_ids[batch_index])
                if own_id in self.route_img_to_index:
                    valid_candidates[self.route_img_to_index[own_id]] = False
            candidate_indices = torch.nonzero(valid_candidates, as_tuple=False).flatten()
            count = min(k, int(candidate_indices.numel()))
            scores = q_route.new_full((k,), -1.0e4)
            valid = torch.zeros(k, device=q_route.device, dtype=torch.bool)
            indices = torch.full((k,), -1, device=q_route.device, dtype=torch.long)
            ids: list[str] = []
            if count > 0:
                candidate_scores = similarities[batch_index].index_select(0, candidate_indices)
                selected_scores, local_indices = torch.topk(candidate_scores, k=count)
                selected_indices = candidate_indices.index_select(0, local_indices)
                scores[:count] = selected_scores
                valid[:count] = True
                indices[:count] = selected_indices
                ids = [image_ids[index] for index in selected_indices.detach().cpu().tolist()]
            score_rows.append(scores)
            valid_rows.append(valid)
            index_rows.append(indices)
            top_ids.append(ids)
        top_scores = torch.stack(score_rows)
        top_valid = torch.stack(valid_rows)
        route_attention = _masked_route_softmax(top_scores, top_valid)
        route_entropy_norm = entropy_from_probs(route_attention, dim=1)
        valid_count = top_valid.sum(dim=1)
        route_entropy = torch.where(
            valid_count > 1,
            route_entropy_norm * valid_count.clamp_min(1).to(q_route.dtype).log(),
            torch.zeros_like(route_entropy_norm),
        )
        return {
            "top_img_ids": top_ids,
            "top_img_scores": top_scores,
            "top_img_valid": top_valid,
            "top_img_indices": torch.stack(index_rows),
            "route_entropy": route_entropy,
            "route_entropy_norm": route_entropy_norm,
        }

    def get_parent_subbank(
        self,
        top_img_ids: Iterable[object] | None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        exclude_image_id: object | None = None,
    ) -> Dict[str, Any]:
        """Transfer only parents belonging to one query's routed images."""

        target_device = torch.device("cpu") if device is None else torch.device(device)
        target_dtype = self.storage_dtype if dtype is None else dtype
        selected_ids = _flatten_image_ids(top_img_ids)
        if exclude_image_id is not None:
            selected_ids = [item for item in selected_ids if item != str(exclude_image_id)]
        if not self.is_ready() or not selected_ids:
            return self._empty_parent_subbank(target_device, target_dtype)
        chunks = [self.parent_img_to_indices[item] for item in selected_ids if item in self.parent_img_to_indices]
        if not chunks:
            return self._empty_parent_subbank(target_device, target_dtype)
        indices = torch.unique(torch.cat(chunks), sorted=True)
        metadata_indices = indices.tolist()
        return {
            "p3_keys": self.parent["p3_keys"].index_select(0, indices).to(
                device=target_device, dtype=target_dtype, non_blocking=True
            ),
            "p3_values": self.parent["p3_values"].index_select(0, indices).to(
                device=target_device, dtype=target_dtype, non_blocking=True
            ),
            "p3_geometry": self.parent["p3_geometry"].index_select(0, indices).to(
                device=target_device, dtype=target_dtype, non_blocking=True
            ),
            "child_ptr": self.parent["child_ptr"].index_select(0, indices).to(
                device=target_device, non_blocking=True
            ),
            "parent_indices": indices.to(device=target_device, non_blocking=True),
            "parent_meta": [self.parent["parent_meta"][index] for index in metadata_indices],
        }

    def get_child_by_ptr(
        self,
        top_child_ptrs: torch.Tensor,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Gather children without ever clamping an invalid ``-1`` pointer."""

        target_device = top_child_ptrs.device if device is None else torch.device(device)
        target_dtype = self.storage_dtype if dtype is None else dtype
        output_shape = tuple(top_child_ptrs.shape)
        keys = torch.zeros((*output_shape, self.memory_dim), device=target_device, dtype=target_dtype)
        geometry = torch.zeros((*output_shape, self.geometry_dim), device=target_device, dtype=target_dtype)
        child_count = int(self.child.get("p2_child_keys", torch.empty(0)).size(0))
        valid = (top_child_ptrs >= 0) & (top_child_ptrs < child_count)
        if valid_mask is not None:
            if valid_mask.shape != top_child_ptrs.shape:
                raise ValueError("valid_mask shape must match top_child_ptrs")
            valid = valid & valid_mask.to(device=valid.device, dtype=torch.bool)
        if child_count == 0 or not bool(valid.any()):
            return {"p2_child_keys": keys, "p2_child_geo": geometry, "child_valid": valid.to(target_device)}
        positions = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
        pointers = top_child_ptrs.reshape(-1).index_select(0, positions).to(device="cpu", dtype=torch.long)
        selected_keys = self.child["p2_child_keys"].index_select(0, pointers).to(
            device=target_device, dtype=target_dtype, non_blocking=True
        )
        selected_geometry = self.child["p2_child_geo"].index_select(0, pointers).to(
            device=target_device, dtype=target_dtype, non_blocking=True
        )
        target_positions = positions.to(device=target_device)
        keys.view(-1, self.memory_dim).index_copy_(0, target_positions, selected_keys)
        geometry.view(-1, self.geometry_dim).index_copy_(0, target_positions, selected_geometry)
        return {
            "p2_child_keys": keys,
            "p2_child_geo": geometry,
            "child_valid": valid.to(device=target_device, non_blocking=True),
        }

    def state_dict(self) -> Dict[str, Any]:
        return {
            "format_version": self.FORMAT_VERSION,
            "schema_version": int(self.compat_meta.get("schema_version", self.DEFAULT_SCHEMA_VERSION)),
            "compat_meta": dict(self.compat_meta),
            "memory_dim": self.memory_dim,
            "value_dim": self.value_dim,
            "geometry_dim": self.geometry_dim,
            "storage_dtype": "float16",
            "route": _cpu_state_copy(self.route),
            "parent": _cpu_state_copy(self.parent),
            "child": _cpu_state_copy(self.child),
            "finalized": bool(self._finalized),
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any] | None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Load a raw memory state or a nested ``{'memory': state}`` checkpoint."""

        self.clear()
        if not state:
            return
        outer = state
        if "memory" in state and isinstance(state["memory"], Mapping):
            state = state["memory"]
        if device is not None and torch.device(device).type != "cpu":
            raise ValueError("Loaded PC-HBM memory must remain on CPU")
        if dtype is not None and dtype != torch.float16:
            raise ValueError("Loaded PC-HBM memory must remain float16")
        for key, expected in (
            ("memory_dim", self.memory_dim),
            ("value_dim", self.value_dim),
            ("geometry_dim", self.geometry_dim),
        ):
            actual = int(state.get(key, expected))
            if actual != expected:
                raise ValueError(f"Memory {key}={actual} is incompatible with expected {expected}")

        outer_meta = dict(outer.get("compat_meta", {}) or {})
        inner_meta = dict(state.get("compat_meta", {}) or {})
        self.compat_meta.update(outer_meta)
        self.compat_meta.update(inner_meta)
        raw_route = state.get("route", {}) or {}
        raw_parent = state.get("parent", {}) or {}
        raw_child = state.get("child", {}) or {}
        self.route = {
            name: _state_float(raw_route.get(name), self.memory_dim)
            for name in (
                "x3_global",
                "x3_boundary",
                "x3_uncertain",
                "x3_bg_near",
                "x3_environment",
                "route_embed",
            )
        }
        self.route["img_ids"] = [str(item) for item in raw_route.get("img_ids", [])]
        self.parent = {
            "p3_keys": _state_float(raw_parent.get("p3_keys"), self.memory_dim),
            "p3_values": _state_float(raw_parent.get("p3_values"), self.value_dim),
            "p3_geometry": _state_float(raw_parent.get("p3_geometry"), self.geometry_dim),
            "child_ptr": _state_long(raw_parent.get("child_ptr")),
            "parent_meta": [dict(item) for item in raw_parent.get("parent_meta", [])],
        }
        self.child = {
            "p2_child_keys": _state_float(raw_child.get("p2_child_keys"), self.memory_dim),
            "p2_child_geo": _state_float(raw_child.get("p2_child_geo"), self.geometry_dim),
            "child_meta": [dict(item) for item in raw_child.get("child_meta", [])],
        }
        self._finalized = bool(state.get("finalized", True))
        self._build_indices()
        self._validate_finalized_storage()

    def diagnostic_string(self) -> str:
        image_count = int(self.route.get("route_embed", torch.empty(0, self.memory_dim)).size(0))
        parent_count = int(self.parent.get("p3_keys", torch.empty(0, self.memory_dim)).size(0))
        child_count = int(self.child.get("p2_child_keys", torch.empty(0, self.memory_dim)).size(0))
        return f"[PC-HBM] images={image_count}, parents={parent_count}, children={child_count}, ready={self.is_ready()}"

    def num_children_pending(self) -> int:
        return sum(int(item.size(0)) for item in self._child_key_list)

    def clear_gpu_cache(self) -> None:
        """Compatibility no-op: this implementation never retains a GPU cache."""

    def _store_float(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().to(device="cpu", dtype=self.storage_dtype).contiguous()

    @staticmethod
    def _check_matrix(tensor: torch.Tensor, width: int, name: str) -> None:
        if tensor.ndim != 2 or tensor.size(1) != int(width):
            raise ValueError(f"{name} must be [N,{width}], got {tuple(tensor.shape)}")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be a floating tensor")

    def _build_indices(self) -> None:
        mapping: dict[str, list[int]] = {}
        for index, metadata in enumerate(self.parent.get("parent_meta", [])):
            image_id = str(metadata.get("image_id", ""))
            if image_id:
                mapping.setdefault(image_id, []).append(index)
        self.parent_img_to_indices = {
            image_id: torch.tensor(indices, dtype=torch.long)
            for image_id, indices in mapping.items()
        }
        self.route_img_to_index = {
            str(image_id): index for index, image_id in enumerate(self.route.get("img_ids", []))
        }

    def _validate_finalized_storage(self) -> None:
        if not self._finalized:
            return
        for group_name, group in (("route", self.route), ("parent", self.parent), ("child", self.child)):
            for name, value in group.items():
                if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                    continue
                if value.device.type != "cpu" or value.dtype != torch.float16:
                    raise ValueError(f"{group_name}.{name} must be a CPU float16 tensor")
        if self.compat_meta.get("source", "labeled_only") != "labeled_only":
            raise ValueError("Loaded memory source is not labeled_only")
        route_count = int(self.route.get("route_embed", torch.empty(0, self.memory_dim)).size(0))
        if route_count != len(self.route.get("img_ids", [])):
            raise ValueError("route embedding/image ID lengths do not match")
        parent_count = int(self.parent.get("p3_keys", torch.empty(0, self.memory_dim)).size(0))
        if parent_count != len(self.parent.get("parent_meta", [])):
            raise ValueError("parent tensor/metadata lengths do not match")
        child_count = int(self.child.get("p2_child_keys", torch.empty(0, self.memory_dim)).size(0))
        if child_count != len(self.child.get("child_meta", [])):
            raise ValueError("child tensor/metadata lengths do not match")

    def _empty_route_result(self, query: torch.Tensor, k: int) -> Dict[str, Any]:
        batch_size = query.size(0)
        return {
            "top_img_ids": [[] for _ in range(batch_size)],
            "top_img_scores": query.new_full((batch_size, k), -1.0e4),
            "top_img_valid": torch.zeros((batch_size, k), device=query.device, dtype=torch.bool),
            "top_img_indices": torch.full((batch_size, k), -1, device=query.device, dtype=torch.long),
            "route_entropy": query.new_zeros(batch_size),
            "route_entropy_norm": query.new_zeros(batch_size),
        }

    def _empty_parent_subbank(self, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        return {
            "p3_keys": torch.empty((0, self.memory_dim), device=device, dtype=dtype),
            "p3_values": torch.empty((0, self.value_dim), device=device, dtype=dtype),
            "p3_geometry": torch.empty((0, self.geometry_dim), device=device, dtype=dtype),
            "child_ptr": torch.empty(0, device=device, dtype=torch.long),
            "parent_indices": torch.empty(0, device=device, dtype=torch.long),
            "parent_meta": [],
        }


PCHBMMemory = PCMemory


def parent_values_from_region(
    region: str,
    sdf: torch.Tensor,
    reliability: torch.Tensor,
) -> torch.Tensor:
    """Build the fixed eight-value parent target for one labelled region."""

    if region not in REGION_TO_ID:
        raise KeyError(f"Unknown PC-HBM region: {region}")
    sdf = sdf.reshape(-1)
    reliability = reliability.reshape(-1)
    if sdf.shape != reliability.shape:
        raise ValueError("sdf and reliability lengths must match")
    values = sdf.new_zeros((sdf.numel(), 8))
    values[:, REGION_TO_ID[region]] = 1.0
    is_foreground = region in {"fg_core", "fg_boundary"}
    values[:, 4] = float(is_foreground)
    values[:, 5] = float(not is_foreground)
    values[:, 6] = sdf
    values[:, 7] = reliability
    return values


def _parse_storage_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if dtype in (torch.float16, "float16", "fp16", "torch.float16"):
        return torch.float16
    raise ValueError(f"Unsupported PC-HBM storage dtype: {dtype}")


def _validate_labeled_meta(metadata: Mapping[str, Any], kind: str) -> dict[str, Any]:
    result = dict(metadata)
    source = str(result.get("source", "labeled_only"))
    if source != "labeled_only" or result.get("is_labeled", True) is False:
        raise ValueError(f"{kind} metadata is not labelled-only")
    if "image_id" not in result:
        raise ValueError(f"{kind} metadata must contain image_id")
    result["image_id"] = str(result["image_id"])
    result["source"] = "labeled_only"
    result["is_labeled"] = True
    return result


def _cat_or_empty(items: Sequence[torch.Tensor], width: int, dtype: torch.dtype) -> torch.Tensor:
    if not items:
        return torch.empty((0, width), dtype=dtype)
    return torch.cat(list(items), dim=0).to(device="cpu", dtype=dtype).contiguous()


def _cat_long_or_empty(items: Sequence[torch.Tensor]) -> torch.Tensor:
    if not items:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(list(items), dim=0).to(device="cpu", dtype=torch.long).contiguous()


def _flatten_image_ids(values: Iterable[object] | None) -> list[str]:
    if values is None:
        return []
    output: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple, set)):
            output.extend(str(item) for item in value)
        elif value is not None:
            output.append(str(value))
    # Preserve route order while removing accidental duplicates.
    return list(dict.fromkeys(output))


def _masked_route_softmax(scores: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    probability = torch.softmax(scores.masked_fill(~valid, -1.0e4), dim=1)
    probability = probability * valid.to(dtype=scores.dtype)
    denominator = probability.sum(dim=1, keepdim=True)
    return torch.where(denominator > 0, probability / denominator.clamp_min(EPS), torch.zeros_like(probability))


def _cpu_state_copy(group: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in group.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu().clone()
        elif isinstance(value, list):
            result[key] = [dict(item) if isinstance(item, Mapping) else item for item in value]
        else:
            result[key] = value
    return result


def _state_float(value: Any, width: int) -> torch.Tensor:
    if value is None:
        return torch.empty((0, width), dtype=torch.float16)
    tensor = torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float16).contiguous()
    if tensor.ndim != 2 or tensor.size(1) != width:
        raise ValueError(f"Stored tensor must be [N,{width}], got {tuple(tensor.shape)}")
    return tensor


def _state_long(value: Any) -> torch.Tensor:
    if value is None:
        return torch.empty(0, dtype=torch.long)
    return torch.as_tensor(value).detach().to(device="cpu", dtype=torch.long).view(-1).contiguous()


def _canonical_meta(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_canonical_meta(item) for item in value)
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _canonical_meta(item)) for key, item in value.items()))
    return value


__all__ = [
    "CompatibilityResult",
    "PCMemory",
    "PCHBMMemory",
    "parent_values_from_region",
]

