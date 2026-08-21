"""CPU-resident, configurable-float labeled Route/Pair memory."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import torch
import torch.nn.functional as F


_ARCHITECTURE = "DINO_SCOD_PC_HBM_LITE"
_REGION_NAMES = ("fg_boundary", "bg_near")
_REQUIRED_META_KEYS = (
    "architecture",
    "schema_version",
    "input_size",
    "token_hw",
    "output_hw",
    "dino_layer_indices",
    "encoder_dim",
    "decoder_dim",
    "memory_dim",
    "child_window_size",
    "region_names",
    "storage_dtype",
    "source",
    "route_environment_min_mass",
    "fg_boundary_kernel",
    "bg_near_kernel",
    "gt_binary_threshold",
    "region_max_quota",
    "region_min_quota",
    "region_sampling_ratio",
)


@dataclass(frozen=True)
class CompatibilityResult:
    """Boolean-compatible compatibility result with an explanatory reason."""

    compatible: bool
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.compatible

    def __iter__(self) -> Iterator[object]:
        yield self.compatible
        yield self.reason


class PCMemory:
    """One-to-one P3/P2 pair store with a separate two-key image route table."""

    FORMAT_VERSION = 2
    DEFAULT_SCHEMA_VERSION = 2

    def __init__(
        self,
        memory_dim: int | None = None,
        *,
        storage_dtype: torch.dtype | str | None = None,
        compat_meta: Mapping[str, Any] | None = None,
        config: Any | None = None,
    ) -> None:
        self.config = config
        if config is not None:
            if not hasattr(config, "expected_memory_meta"):
                raise TypeError("config must provide expected_memory_meta()")
            config_meta = dict(config.expected_memory_meta())
            config_memory_dim = int(config_meta["memory_dim"])
            if memory_dim is not None and int(memory_dim) != config_memory_dim:
                raise ValueError(
                    "memory_dim must match config.memory_dim: "
                    f"{memory_dim} != {config_memory_dim}"
                )
            resolved_memory_dim = config_memory_dim
            config_dtype = config_meta["storage_dtype"]
            if (
                storage_dtype is not None
                and _parse_storage_dtype(storage_dtype)
                != _parse_storage_dtype(config_dtype)
            ):
                raise ValueError(
                    "storage_dtype must match config.memory_storage_dtype"
                )
            resolved_storage_dtype = config_dtype
            expected_meta = config_meta
        else:
            resolved_memory_dim = 128 if memory_dim is None else int(memory_dim)
            resolved_storage_dtype = (
                torch.float16 if storage_dtype is None else storage_dtype
            )
            expected_meta = _default_compat_meta(
                memory_dim=resolved_memory_dim,
                storage_dtype=_storage_dtype_name(
                    _parse_storage_dtype(resolved_storage_dtype)
                ),
            )
        self.memory_dim = int(resolved_memory_dim)
        if self.memory_dim < 1:
            raise ValueError("memory_dim must be positive")
        self.storage_dtype = _parse_storage_dtype(resolved_storage_dtype)
        self.region_names = tuple(expected_meta["region_names"])
        if (
            len(self.region_names) != 2
            or any(not isinstance(name, str) or not name for name in self.region_names)
            or len(set(self.region_names)) != 2
        ):
            raise ValueError("region_names must contain two unique non-empty strings")
        self._expected_compat_meta = dict(expected_meta)
        initial_meta = dict(expected_meta)
        initial_meta.update(dict(compat_meta or {}))
        self._initial_compat_meta = initial_meta
        self.clear()

    def clear(self) -> None:
        self._route_global_list: list[torch.Tensor] = []
        self._route_environment_list: list[torch.Tensor] = []
        self._route_img_ids: list[str] = []
        self._pair_p3_list: list[torch.Tensor] = []
        self._pair_p2_list: list[torch.Tensor] = []
        self._pair_region_list: list[torch.Tensor] = []
        self._pair_meta_list: list[dict[str, Any]] = []
        self.route: dict[str, Any] = {}
        self.pairs: dict[str, Any] = {}
        self.compat_meta: dict[str, Any] = dict(self._initial_compat_meta)
        self.route_img_to_index: dict[str, int] = {}
        self.pair_img_to_indices: dict[str, torch.Tensor] = {}
        self._finalized = False

    def append(
        self,
        entries: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    ) -> None:
        """Append one builder result or a sequence of labeled builder results."""

        self._ensure_mutable()
        if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes, Mapping)):
            for item in entries:
                self.append(item)
            return
        if not isinstance(entries, Mapping):
            raise TypeError("memory entries must be a mapping or sequence of mappings")
        if str(entries.get("source", "labeled_only")) != "labeled_only":
            raise ValueError("PC-HBM-Lite memory accepts labeled data only")
        if str(entries.get("producer_role", "labeled_student")) != "labeled_student":
            raise ValueError("PC-HBM-Lite memory accepts labeled_student producer only")
        unknown = set(entries).difference(
            {"source", "producer_role", "route", "pairs", "compat_meta"}
        )
        if unknown:
            raise ValueError(f"Unknown memory entry groups: {sorted(unknown)}")

        route = entries.get("route")
        if route is not None:
            if not isinstance(route, Mapping):
                raise TypeError("route entries must be a mapping")
            expected_route_keys = {
                "global_keys",
                "environment_keys",
                "img_ids",
            }
            if set(route) != expected_route_keys:
                raise ValueError(
                    "Route entries must contain exactly "
                    f"{sorted(expected_route_keys)}, got {sorted(route)}"
                )
        pairs = entries.get("pairs")
        if pairs is not None:
            if not isinstance(pairs, Mapping):
                raise TypeError("pair entries must be a mapping")
            expected_pair_keys = {
                "p3_keys",
                "p2_keys",
                "region_ids",
                "pair_meta",
            }
            if set(pairs) != expected_pair_keys:
                raise ValueError(
                    "Pair entries must contain exactly "
                    f"{sorted(expected_pair_keys)}, got {sorted(pairs)}"
                )

        if "compat_meta" in entries:
            self.compat_meta.update(dict(entries["compat_meta"] or {}))
        if route is not None:
            self.append_route(
                global_keys=route["global_keys"],
                environment_keys=route["environment_keys"],
                img_ids=route["img_ids"],
            )
        if pairs is not None:
            self.append_pairs(
                p3_keys=pairs["p3_keys"],
                p2_keys=pairs["p2_keys"],
                region_ids=pairs["region_ids"],
                pair_meta=pairs["pair_meta"],
            )

    def append_route(
        self,
        *,
        global_keys: torch.Tensor,
        environment_keys: torch.Tensor,
        img_ids: Sequence[object],
    ) -> None:
        self._ensure_mutable()
        self._check_matrix(global_keys, "global_keys")
        self._check_matrix(environment_keys, "environment_keys")
        if global_keys.shape != environment_keys.shape:
            raise ValueError("global_keys and environment_keys must have identical shapes")
        normalized_ids = _normalize_image_ids(img_ids)
        if len(normalized_ids) != global_keys.size(0):
            raise ValueError("Route key count must match img_ids")
        duplicates = set(self._route_img_ids).intersection(normalized_ids)
        if duplicates:
            raise ValueError(f"Duplicate image IDs in labeled memory: {sorted(duplicates)}")
        self._route_global_list.append(self._store_float(global_keys))
        self._route_environment_list.append(self._store_float(environment_keys))
        self._route_img_ids.extend(normalized_ids)

    def append_pairs(
        self,
        *,
        p3_keys: torch.Tensor,
        p2_keys: torch.Tensor,
        region_ids: torch.Tensor,
        pair_meta: Sequence[Mapping[str, Any]],
    ) -> None:
        self._ensure_mutable()
        self._check_matrix(p3_keys, "p3_keys")
        self._check_matrix(p2_keys, "p2_keys")
        if p3_keys.shape != p2_keys.shape:
            raise ValueError("P3 and P2 pair keys must be one-to-one")
        regions = torch.as_tensor(region_ids).detach().to(
            device="cpu",
            dtype=torch.long,
        ).view(-1).contiguous()
        count = int(p3_keys.size(0))
        if regions.numel() != count or len(pair_meta) != count:
            raise ValueError("Pair keys, region IDs, and metadata lengths must match")
        if regions.numel() and not bool(((regions == 0) | (regions == 1)).all()):
            raise ValueError("Pair region IDs must be 0 or 1")
        normalized_meta = [
            _normalize_pair_meta(
                metadata,
                int(region_id),
                self.region_names,
            )
            for metadata, region_id in zip(pair_meta, regions.tolist())
        ]
        self._pair_p3_list.append(self._store_float(p3_keys))
        self._pair_p2_list.append(self._store_float(p2_keys))
        self._pair_region_list.append(regions)
        self._pair_meta_list.extend(normalized_meta)

    def finalize(
        self,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
        *,
        compat_meta: Mapping[str, Any] | None = None,
    ) -> None:
        device = torch.device(device)
        resolved_dtype = self.storage_dtype if dtype is None else dtype
        if device.type != "cpu" or resolved_dtype != self.storage_dtype:
            raise ValueError(
                "PC-HBM-Lite memory must be finalized on CPU using its "
                f"configured storage dtype {_storage_dtype_name(self.storage_dtype)}"
            )
        route = {
            "global_keys": _cat_float(
                self._route_global_list,
                self.memory_dim,
                self.storage_dtype,
            ),
            "environment_keys": _cat_float(
                self._route_environment_list,
                self.memory_dim,
                self.storage_dtype,
            ),
            "img_ids": list(self._route_img_ids),
        }
        pairs = {
            "p3_keys": _cat_float(
                self._pair_p3_list,
                self.memory_dim,
                self.storage_dtype,
            ),
            "p2_keys": _cat_float(
                self._pair_p2_list,
                self.memory_dim,
                self.storage_dtype,
            ),
            "region_ids": _cat_long(self._pair_region_list),
            "pair_meta": [dict(item) for item in self._pair_meta_list],
        }
        meta = dict(self.compat_meta)
        meta.update(dict(compat_meta or {}))
        _validate_meta(meta, self._expected_compat_meta)
        _validate_tables(
            route,
            pairs,
            self.memory_dim,
            self.region_names,
        )

        self.route = route
        self.pairs = pairs
        self.compat_meta = meta
        self._finalized = True
        self._build_indices()
        self._validate_storage()

    def is_ready(self) -> bool:
        if not self._finalized:
            return False
        route_count = int(self.route.get("global_keys", torch.empty(0, self.memory_dim)).size(0))
        region_ids = self.pairs.get("region_ids", torch.empty(0, dtype=torch.long))
        return bool(
            route_count > 0
            and region_ids.numel() > 0
            and (region_ids == 0).any()
            and (region_ids == 1).any()
        )

    def validate_compat(
        self,
        expected: Mapping[str, Any] | object | None,
        *,
        require_producer_match: bool = False,
    ) -> CompatibilityResult:
        if not self._finalized:
            return CompatibilityResult(False, "memory_not_finalized")
        try:
            _validate_meta(self.compat_meta, self._expected_compat_meta)
        except (TypeError, ValueError) as exc:
            return CompatibilityResult(False, str(exc))

        if expected is None:
            expected_meta = dict(self._expected_compat_meta)
        elif isinstance(expected, Mapping):
            expected_meta = dict(expected)
        elif hasattr(expected, "expected_memory_meta"):
            expected_meta = dict(expected.expected_memory_meta())
        else:
            return CompatibilityResult(False, "invalid_expected_compatibility")

        for key, expected_value in expected_meta.items():
            if key == "producer_fingerprint" and not require_producer_match:
                continue
            if key not in self.compat_meta:
                return CompatibilityResult(False, f"compat_mismatch:{key}")
            if _canonical(self.compat_meta[key]) != _canonical(expected_value):
                return CompatibilityResult(False, f"compat_mismatch:{key}")
        if require_producer_match:
            fingerprint = expected_meta.get("producer_fingerprint")
            if fingerprint is None or self.compat_meta.get("producer_fingerprint") != fingerprint:
                return CompatibilityResult(False, "compat_mismatch:producer_fingerprint")
        return CompatibilityResult(True, None)

    def route_query(
        self,
        q_global: torch.Tensor,
        q_environment: torch.Tensor,
        top_img_k: int,
        *,
        query_image_ids: Sequence[object] | None = None,
        exclude_self_match: bool = True,
        global_weight: float | None = None,
        environment_weight: float | None = None,
    ) -> dict[str, Any]:
        """Rank labeled images by separate global/environment cosine scores."""

        self._check_query_pair(q_global, q_environment)
        k = int(top_img_k)
        if k <= 0:
            raise ValueError("top_img_k must be positive")
        if query_image_ids is not None and len(query_image_ids) != q_global.size(0):
            raise ValueError("query_image_ids length must match query batch size")
        if not self.is_ready():
            return self._empty_route_result(q_global, k)

        query_global = F.normalize(
            torch.nan_to_num(q_global.float()),
            dim=-1,
            eps=1.0e-6,
        )
        query_environment = F.normalize(
            torch.nan_to_num(q_environment.float()),
            dim=-1,
            eps=1.0e-6,
        )
        memory_global = F.normalize(
            self.route["global_keys"].to(
                device=q_global.device,
                dtype=torch.float32,
                non_blocking=True,
            ),
            dim=-1,
            eps=1.0e-6,
        )
        memory_environment = F.normalize(
            self.route["environment_keys"].to(
                device=q_global.device,
                dtype=torch.float32,
                non_blocking=True,
            ),
            dim=-1,
            eps=1.0e-6,
        )
        global_scores = query_global @ memory_global.transpose(0, 1)
        environment_scores = query_environment @ memory_environment.transpose(0, 1)
        raw_global_weight = (
            getattr(self.config, "route_global_weight", 0.5)
            if global_weight is None
            else global_weight
        )
        raw_environment_weight = (
            getattr(self.config, "route_environment_weight", 0.5)
            if environment_weight is None
            else environment_weight
        )
        normalized_global, normalized_environment = _normalize_route_weights(
            raw_global_weight,
            raw_environment_weight,
        )
        combined = (
            normalized_global * global_scores
            + normalized_environment * environment_scores
        )
        valid = torch.ones_like(combined, dtype=torch.bool)

        if exclude_self_match and query_image_ids is not None:
            for batch_index, raw_image_id in enumerate(query_image_ids):
                route_index = self.route_img_to_index.get(str(raw_image_id))
                if route_index is not None:
                    valid[batch_index, route_index] = False
        ranked = torch.argsort(
            combined.masked_fill(~valid, -1.0e4),
            dim=1,
            descending=True,
            stable=True,
        )
        real_k = min(k, combined.size(1))
        selected = ranked[:, :real_k]
        selected_valid = valid.gather(1, selected)
        selected_scores = combined.gather(1, selected).masked_fill(~selected_valid, -1.0e4)

        batch_size = q_global.size(0)
        top_scores = torch.full(
            (batch_size, k),
            -1.0e4,
            device=q_global.device,
            dtype=torch.float32,
        )
        top_valid = torch.zeros((batch_size, k), device=q_global.device, dtype=torch.bool)
        top_indices = torch.full(
            (batch_size, k),
            -1,
            device=q_global.device,
            dtype=torch.long,
        )
        if real_k:
            top_scores[:, :real_k] = selected_scores
            top_valid[:, :real_k] = selected_valid
            top_indices[:, :real_k] = selected.masked_fill(~selected_valid, -1)

        top_img_ids: list[list[str]] = []
        route_ids = self.route["img_ids"]
        for row_indices, row_valid in zip(top_indices.tolist(), top_valid.tolist()):
            top_img_ids.append(
                [
                    route_ids[index]
                    for index, item_valid in zip(row_indices, row_valid)
                    if item_valid
                ]
            )
        route_entropy_norm = _normalized_masked_entropy(top_scores, top_valid)
        return {
            "top_img_ids": top_img_ids,
            "top_img_scores": top_scores,
            "top_img_valid": top_valid,
            "top_img_indices": top_indices,
            "route_entropy_norm": route_entropy_norm,
        }

    def get_pair_subbank(
        self,
        top_img_ids: Iterable[object] | None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        exclude_image_id: object | None = None,
    ) -> dict[str, Any]:
        target_device = torch.device("cpu" if device is None else device)
        target_dtype = self.storage_dtype if dtype is None else dtype
        if not torch.empty((), dtype=target_dtype).is_floating_point():
            raise TypeError("Pair key dtype must be floating point")
        if not self._finalized:
            return self._empty_pair_subbank(target_device, target_dtype)

        requested = (
            list(self.route.get("img_ids", []))
            if top_img_ids is None
            else _flatten_image_ids(top_img_ids)
        )
        excluded = None if exclude_image_id is None else str(exclude_image_id)
        selected_parts = [
            self.pair_img_to_indices[image_id]
            for image_id in requested
            if image_id != excluded and image_id in self.pair_img_to_indices
        ]
        if not selected_parts:
            return self._empty_pair_subbank(target_device, target_dtype)
        pair_indices_cpu = torch.cat(selected_parts).unique(sorted=True)
        metadata = [
            dict(self.pairs["pair_meta"][index])
            for index in pair_indices_cpu.tolist()
        ]
        return {
            "p3_keys": self.pairs["p3_keys"].index_select(0, pair_indices_cpu).to(
                device=target_device,
                dtype=target_dtype,
                non_blocking=True,
            ),
            "p2_keys": self.pairs["p2_keys"].index_select(0, pair_indices_cpu).to(
                device=target_device,
                dtype=target_dtype,
                non_blocking=True,
            ),
            "region_ids": self.pairs["region_ids"].index_select(0, pair_indices_cpu).to(
                device=target_device,
                dtype=torch.long,
                non_blocking=True,
            ),
            "pair_indices": pair_indices_cpu.to(
                device=target_device,
                dtype=torch.long,
                non_blocking=True,
            ),
            "pair_meta": metadata,
        }

    def state_dict(self) -> dict[str, Any]:
        if not self._finalized:
            raise RuntimeError("Cannot export a non-finalized PC-HBM-Lite memory")
        return {
            "format_version": self.FORMAT_VERSION,
            "schema_version": self.DEFAULT_SCHEMA_VERSION,
            "compat_meta": dict(self.compat_meta),
            "memory_dim": self.memory_dim,
            "storage_dtype": _storage_dtype_name(self.storage_dtype),
            "route": _cpu_group_copy(self.route),
            "pairs": _cpu_group_copy(self.pairs),
            "finalized": True,
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Atomically validate and load only a finalized Schema V2 state."""

        if not isinstance(state, Mapping):
            raise TypeError("PC-HBM-Lite state must be a mapping")
        raw = state.get("memory", state)
        if not isinstance(raw, Mapping):
            raise TypeError("Nested PC-HBM-Lite state must be a mapping")
        if device is not None and torch.device(device).type != "cpu":
            raise ValueError("Loaded PC-HBM-Lite memory must remain on CPU")
        if dtype is not None and dtype != self.storage_dtype:
            raise ValueError(
                "Loaded PC-HBM-Lite memory dtype must match the configured "
                f"{_storage_dtype_name(self.storage_dtype)} storage"
            )

        required = {
            "format_version",
            "schema_version",
            "compat_meta",
            "memory_dim",
            "storage_dtype",
            "route",
            "pairs",
            "finalized",
        }
        if int(raw.get("format_version", -1)) != self.FORMAT_VERSION:
            raise ValueError("Incompatible PC-HBM memory: compat_mismatch:schema_version")
        if int(raw.get("schema_version", -1)) != self.DEFAULT_SCHEMA_VERSION:
            raise ValueError("Incompatible PC-HBM memory: compat_mismatch:schema_version")
        missing = required.difference(raw)
        unknown = set(raw).difference(required)
        if missing or unknown:
            raise ValueError(
                "Incompatible PC-HBM memory: invalid_structure:"
                f"missing={sorted(missing)},unknown={sorted(unknown)}"
            )
        if not bool(raw["finalized"]):
            raise ValueError("Incompatible PC-HBM memory: memory_not_finalized")
        if int(raw["memory_dim"]) != self.memory_dim:
            raise ValueError("Incompatible PC-HBM memory: compat_mismatch:memory_dim")
        if raw["storage_dtype"] != _storage_dtype_name(self.storage_dtype):
            raise ValueError(
                "Incompatible PC-HBM memory: compat_mismatch:storage_dtype"
            )

        meta = dict(raw["compat_meta"])
        _validate_meta(meta, self._expected_compat_meta)
        route = _load_route_table(
            raw["route"],
            self.memory_dim,
            self.storage_dtype,
        )
        pairs = _load_pair_table(
            raw["pairs"],
            self.memory_dim,
            self.storage_dtype,
            self.region_names,
        )
        _validate_tables(
            route,
            pairs,
            self.memory_dim,
            self.region_names,
        )

        # Commit only after the entire incoming state has passed preflight.
        self._route_global_list = []
        self._route_environment_list = []
        self._route_img_ids = []
        self._pair_p3_list = []
        self._pair_p2_list = []
        self._pair_region_list = []
        self._pair_meta_list = []
        self.route = route
        self.pairs = pairs
        self.compat_meta = meta
        self._finalized = True
        self._build_indices()
        self._validate_storage()

    def diagnostic_string(self) -> str:
        image_count = int(self.route.get("global_keys", torch.empty(0, self.memory_dim)).size(0))
        pair_count = int(self.pairs.get("p3_keys", torch.empty(0, self.memory_dim)).size(0))
        return (
            f"[PC-HBM-Lite] images={image_count}, pairs={pair_count}, "
            f"ready={self.is_ready()}"
        )

    def _ensure_mutable(self) -> None:
        if self._finalized:
            raise RuntimeError("Clear finalized PC-HBM-Lite memory before appending")

    def _store_float(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.detach().to(
            device="cpu",
            dtype=self.storage_dtype,
        ).contiguous()

    def _check_matrix(self, tensor: torch.Tensor, name: str) -> None:
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise TypeError(f"{name} must be a floating tensor")
        if tensor.ndim != 2 or tensor.size(1) != self.memory_dim:
            raise ValueError(
                f"{name} must be [N,{self.memory_dim}], got {tuple(tensor.shape)}"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} contains NaN/Inf")
        if tensor.numel() and bool(
            (
                tensor.detach().float().abs()
                > torch.finfo(self.storage_dtype).max
            ).any()
        ):
            raise ValueError(
                f"{name} exceeds the finite "
                f"{_storage_dtype_name(self.storage_dtype)} range"
            )

    def _check_query_pair(
        self,
        q_global: torch.Tensor,
        q_environment: torch.Tensor,
    ) -> None:
        self._check_matrix(q_global, "q_global")
        self._check_matrix(q_environment, "q_environment")
        if q_global.shape != q_environment.shape:
            raise ValueError("Route query contexts must have identical shapes")
        if q_global.device != q_environment.device:
            raise ValueError("Route query contexts must share a device")

    def _build_indices(self) -> None:
        self.route_img_to_index = {
            str(image_id): index
            for index, image_id in enumerate(self.route.get("img_ids", []))
        }
        by_image: dict[str, list[int]] = {}
        for index, metadata in enumerate(self.pairs.get("pair_meta", [])):
            image_id = str(metadata["image_id"])
            by_image.setdefault(image_id, []).append(index)
        self.pair_img_to_indices = {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in by_image.items()
        }

    def _validate_storage(self) -> None:
        for group_name, group in (("route", self.route), ("pairs", self.pairs)):
            for name, value in group.items():
                if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                    continue
                if (
                    value.device.type != "cpu"
                    or value.dtype != self.storage_dtype
                    or not value.is_contiguous()
                ):
                    raise ValueError(
                        f"{group_name}.{name} must be contiguous CPU "
                        f"{_storage_dtype_name(self.storage_dtype)}"
                    )
                if not bool(torch.isfinite(value).all()):
                    raise ValueError(f"{group_name}.{name} contains NaN/Inf")

    def _empty_route_result(self, query: torch.Tensor, k: int) -> dict[str, Any]:
        batch_size = query.size(0)
        return {
            "top_img_ids": [[] for _ in range(batch_size)],
            "top_img_scores": torch.full(
                (batch_size, k),
                -1.0e4,
                device=query.device,
                dtype=torch.float32,
            ),
            "top_img_valid": torch.zeros(
                (batch_size, k),
                device=query.device,
                dtype=torch.bool,
            ),
            "top_img_indices": torch.full(
                (batch_size, k),
                -1,
                device=query.device,
                dtype=torch.long,
            ),
            "route_entropy_norm": torch.zeros(
                batch_size,
                device=query.device,
                dtype=torch.float32,
            ),
        }

    def _empty_pair_subbank(
        self,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, Any]:
        return {
            "p3_keys": torch.empty((0, self.memory_dim), device=device, dtype=dtype),
            "p2_keys": torch.empty((0, self.memory_dim), device=device, dtype=dtype),
            "region_ids": torch.empty(0, device=device, dtype=torch.long),
            "pair_indices": torch.empty(0, device=device, dtype=torch.long),
            "pair_meta": [],
        }


def _default_compat_meta(
    *,
    memory_dim: int = 128,
    storage_dtype: str = "float16",
) -> dict[str, Any]:
    return {
        "architecture": _ARCHITECTURE,
        "schema_version": 2,
        "input_size": 392,
        "token_hw": (28, 28),
        "output_hw": (98, 98),
        "dino_layer_indices": (2, 5, 8, 11),
        "encoder_dim": 768,
        "decoder_dim": 128,
        "memory_dim": int(memory_dim),
        "child_window_size": 3,
        "region_names": _REGION_NAMES,
        "storage_dtype": str(storage_dtype),
        "source": "labeled_only",
        "producer_role": "labeled_student",
        "route_environment_min_mass": 1.0e-3,
        "fg_boundary_kernel": 3,
        "bg_near_kernel": 7,
        "gt_binary_threshold": 0.5,
        "region_max_quota": (48, 48),
        "region_min_quota": (8, 8),
        "region_sampling_ratio": (0.5, 0.5),
    }


def _parse_storage_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if dtype in (torch.float16, "float16", "fp16", "torch.float16"):
        return torch.float16
    if dtype in (torch.bfloat16, "bfloat16", "bf16", "torch.bfloat16"):
        return torch.bfloat16
    if dtype in (torch.float32, "float32", "fp32", "torch.float32"):
        return torch.float32
    raise ValueError(f"Unsupported PC-HBM-Lite storage dtype: {dtype}")


def _storage_dtype_name(dtype: torch.dtype | str) -> str:
    resolved = _parse_storage_dtype(dtype)
    return {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }[resolved]


def _normalize_image_ids(values: Sequence[object]) -> list[str]:
    normalized = [str(value) for value in values]
    if any(not value for value in normalized):
        raise ValueError("Memory image IDs must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError("Duplicate image IDs within one route append are not allowed")
    return normalized


def _normalize_pair_meta(
    metadata: Mapping[str, Any],
    region_id: int,
    region_names: Sequence[str] = _REGION_NAMES,
) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError("Each pair metadata item must be a mapping")
    result = dict(metadata)
    if "image_id" not in result or not str(result["image_id"]):
        raise ValueError("Pair metadata must contain a non-empty image_id")
    if str(result.get("source", "labeled_only")) != "labeled_only":
        raise ValueError("Pair metadata must come from labeled data")
    if result.get("is_labeled", True) is False:
        raise ValueError("Pair metadata must come from labeled data")
    if len(region_names) != 2:
        raise ValueError("region_names must contain exactly two entries")
    expected_name = str(region_names[int(region_id)])
    if "region_id" in result and int(result["region_id"]) != int(region_id):
        raise ValueError("Pair metadata region ID does not match its tensor")
    if "region" in result and str(result["region"]) != expected_name:
        raise ValueError("Pair metadata region name does not match its tensor")
    result.update(
        {
            "image_id": str(result["image_id"]),
            "region_id": int(region_id),
            "region": expected_name,
            "source": "labeled_only",
            "is_labeled": True,
        }
    )
    return result


def _cat_float(
    items: Sequence[torch.Tensor],
    width: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not items:
        return torch.empty((0, width), dtype=dtype)
    return torch.cat(list(items), dim=0).to(
        device="cpu",
        dtype=dtype,
    ).contiguous()


def _cat_long(items: Sequence[torch.Tensor]) -> torch.Tensor:
    if not items:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(list(items), dim=0).to(
        device="cpu",
        dtype=torch.long,
    ).contiguous()


def _validate_meta(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if not isinstance(actual, Mapping):
        raise TypeError("Incompatible PC-HBM memory: compat_meta_not_mapping")
    allowed = set(_REQUIRED_META_KEYS) | {
        "producer_fingerprint",
        "producer_role",
    }
    unknown = set(actual).difference(allowed)
    if unknown:
        raise ValueError(
            "Incompatible PC-HBM memory: invalid_compat_meta:"
            f"unknown={sorted(unknown)}"
        )
    for key in _REQUIRED_META_KEYS:
        if key not in actual:
            raise ValueError(f"Incompatible PC-HBM memory: compat_mismatch:{key}")
    fingerprint = actual.get("producer_fingerprint")
    if fingerprint is not None and (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
    ):
        raise ValueError(
            "Incompatible PC-HBM memory: compat_mismatch:producer_fingerprint"
        )
    for key, expected_value in expected.items():
        if key == "producer_fingerprint":
            continue
        if key not in actual or _canonical(actual[key]) != _canonical(expected_value):
            raise ValueError(f"Incompatible PC-HBM memory: compat_mismatch:{key}")


def _validate_tables(
    route: Mapping[str, Any],
    pairs: Mapping[str, Any],
    memory_dim: int,
    region_names: Sequence[str] = _REGION_NAMES,
) -> None:
    route_keys = {"global_keys", "environment_keys", "img_ids"}
    pair_keys = {"p3_keys", "p2_keys", "region_ids", "pair_meta"}
    if set(route) != route_keys or set(pairs) != pair_keys:
        raise ValueError("Incompatible PC-HBM memory: invalid_table_structure")
    global_keys = route["global_keys"]
    environment_keys = route["environment_keys"]
    if (
        global_keys.ndim != 2
        or global_keys.size(1) != memory_dim
        or global_keys.shape != environment_keys.shape
        or global_keys.size(0) != len(route["img_ids"])
    ):
        raise ValueError("Incompatible PC-HBM memory: invalid_route_shape")
    route_ids = _normalize_image_ids(route["img_ids"])
    if route_ids != list(route["img_ids"]):
        raise ValueError("Incompatible PC-HBM memory: noncanonical_image_ids")

    p3_keys = pairs["p3_keys"]
    p2_keys = pairs["p2_keys"]
    region_ids = pairs["region_ids"]
    pair_meta = pairs["pair_meta"]
    if (
        p3_keys.ndim != 2
        or p3_keys.size(1) != memory_dim
        or p3_keys.shape != p2_keys.shape
        or p3_keys.size(0) != region_ids.numel()
        or p3_keys.size(0) != len(pair_meta)
    ):
        raise ValueError("Incompatible PC-HBM memory: invalid_pair_shape")
    if region_ids.numel() and not bool(((region_ids == 0) | (region_ids == 1)).all()):
        raise ValueError("Incompatible PC-HBM memory: invalid_region_ids")
    route_id_set = set(route_ids)
    for metadata, region_id in zip(pair_meta, region_ids.tolist()):
        normalized = _normalize_pair_meta(
            metadata,
            int(region_id),
            region_names,
        )
        if normalized["image_id"] not in route_id_set:
            raise ValueError("Pair image IDs must be present in the route table")


def _load_route_table(
    raw: Any,
    memory_dim: int,
    storage_dtype: torch.dtype,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError("Incompatible PC-HBM memory: route_not_mapping")
    if set(raw) != {"global_keys", "environment_keys", "img_ids"}:
        raise ValueError("Incompatible PC-HBM memory: invalid_route_structure")
    raw_img_ids = raw["img_ids"]
    if not isinstance(raw_img_ids, list):
        raise TypeError("Incompatible PC-HBM memory: img_ids must be list[str]")
    return {
        "global_keys": _load_float_matrix(
            raw["global_keys"],
            memory_dim,
            storage_dtype,
        ),
        "environment_keys": _load_float_matrix(
            raw["environment_keys"],
            memory_dim,
            storage_dtype,
        ),
        "img_ids": _normalize_image_ids(raw_img_ids),
    }


def _load_pair_table(
    raw: Any,
    memory_dim: int,
    storage_dtype: torch.dtype,
    region_names: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise TypeError("Incompatible PC-HBM memory: pairs_not_mapping")
    if set(raw) != {"p3_keys", "p2_keys", "region_ids", "pair_meta"}:
        raise ValueError("Incompatible PC-HBM memory: invalid_pair_structure")
    p3_keys = _load_float_matrix(
        raw["p3_keys"],
        memory_dim,
        storage_dtype,
    )
    p2_keys = _load_float_matrix(
        raw["p2_keys"],
        memory_dim,
        storage_dtype,
    )
    raw_region_ids = torch.as_tensor(raw["region_ids"])
    if raw_region_ids.ndim != 1 or raw_region_ids.dtype != torch.long:
        raise ValueError(
            "Incompatible PC-HBM memory: region_ids must be a LongTensor[N]"
        )
    region_ids = raw_region_ids.detach().cpu().contiguous()
    raw_meta = raw["pair_meta"]
    if not isinstance(raw_meta, list):
        raise TypeError("Incompatible PC-HBM memory: pair_meta must be list[dict]")
    if len(raw_meta) != region_ids.numel():
        raise ValueError("Incompatible PC-HBM memory: invalid_pair_shape")
    pair_meta = [
        _normalize_pair_meta(
            metadata,
            int(region_id),
            region_names,
        )
        for metadata, region_id in zip(raw_meta, region_ids.tolist())
    ]
    return {
        "p3_keys": p3_keys,
        "p2_keys": p2_keys,
        "region_ids": region_ids,
        "pair_meta": pair_meta,
    }


def _load_float_matrix(
    value: Any,
    width: int,
    storage_dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not tensor.is_floating_point():
        raise TypeError(
            "Incompatible PC-HBM memory: pair/route keys must be floating"
        )
    if tensor.ndim != 2 or tensor.size(1) != width:
        raise ValueError(
            f"Incompatible PC-HBM memory: expected [N,{width}], got {tuple(tensor.shape)}"
        )
    if (
        tensor.device.type != "cpu"
        or tensor.dtype != storage_dtype
        or not tensor.is_contiguous()
    ):
        raise ValueError(
            "Incompatible PC-HBM memory: pair/route keys must be contiguous "
            f"CPU {_storage_dtype_name(storage_dtype)}"
        )
    work = tensor.detach().float()
    if not bool(torch.isfinite(work).all()):
        raise ValueError(
            "Incompatible PC-HBM memory: pair/route keys contain NaN/Inf"
        )
    if work.numel() and bool(
        (work.abs() > torch.finfo(storage_dtype).max).any()
    ):
        raise ValueError(
            "Incompatible PC-HBM memory: pair/route keys overflow "
            f"{_storage_dtype_name(storage_dtype)}"
        )
    return tensor.detach().clone()


def _flatten_image_ids(values: Iterable[object]) -> list[str]:
    output: list[str] = []
    for value in values:
        if isinstance(value, (list, tuple, set)):
            output.extend(str(item) for item in value)
        elif value is not None:
            output.append(str(value))
    return list(dict.fromkeys(output))


def _normalized_masked_entropy(
    scores: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    masked = scores.masked_fill(~valid, -1.0e4)
    probability = torch.softmax(masked.float(), dim=1)
    probability = probability * valid.float()
    probability = torch.where(
        probability.sum(dim=1, keepdim=True) > 0,
        probability / probability.sum(dim=1, keepdim=True).clamp_min(1.0e-8),
        torch.zeros_like(probability),
    )
    entropy = -(probability * probability.clamp_min(1.0e-8).log()).sum(dim=1)
    count = valid.sum(dim=1)
    denominator = count.clamp_min(2).float().log()
    normalized = torch.where(count > 1, entropy / denominator, torch.zeros_like(entropy))
    return torch.nan_to_num(normalized).clamp(0.0, 1.0)


def _normalize_route_weights(
    global_weight: object,
    environment_weight: object,
) -> tuple[float, float]:
    global_value = float(global_weight)
    environment_value = float(environment_weight)
    if (
        not math.isfinite(global_value)
        or not math.isfinite(environment_value)
        or global_value < 0.0
        or environment_value < 0.0
        or max(global_value, environment_value) <= 0.0
    ):
        raise ValueError(
            "route weights must be finite, non-negative, and not both zero"
        )
    scale = max(global_value, environment_value)
    scaled_global = global_value / scale
    scaled_environment = environment_value / scale
    scaled_total = scaled_global + scaled_environment
    return (
        scaled_global / scaled_total,
        scaled_environment / scaled_total,
    )


def _cpu_group_copy(group: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in group.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu().clone()
        elif isinstance(value, list):
            result[key] = [
                dict(item) if isinstance(item, Mapping) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


def _canonical(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(item) for item in value)
    if isinstance(value, Mapping):
        return tuple(
            sorted((str(key), _canonical(item)) for key, item in value.items())
        )
    return value


__all__ = ["CompatibilityResult", "PCMemory"]
