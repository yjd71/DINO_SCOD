from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Mapping, Sequence

import torch


ENCODER_PC_MEMORY_ARCHITECTURE = "DINO_SCOD_ENCODER_PC_HBM"
ENCODER_PC_MEMORY_SCHEMA_VERSION = 3
ENCODER_PC_MEMORY_FORMAT_VERSION = 3
ENCODER_PC_ROUTE_SOURCES = (
    "encoder_route_key_v1",
    "route_mlp_640_to_128_v1",
    (
        "block11_cls",
        "block11_f4_global",
        "block8_f3_boundary",
        "block8_f3_uncertainty",
        "block8_f3_environment",
    ),
)

ENCODER_PC_STATIC_COMPAT_META: dict[str, Any] = {
    "architecture": ENCODER_PC_MEMORY_ARCHITECTURE,
    "schema_version": ENCODER_PC_MEMORY_SCHEMA_VERSION,
    "adapter_architecture": "encoder_pc_hbm_v1",
    "feature_space": "frozen_dinov2_projected_encoder_v1",
    "route_source": ENCODER_PC_ROUTE_SOURCES,
    "parent_source": "block8_patch",
    "child_source": "block5_patch",
    "detail_source": "block2_patch",
    "input_size": 392,
    "token_hw": (28, 28),
    "dino_layer_indices": (2, 5, 8, 11),
    "dino_checkpoint": "weight/dinov2_vitb14_pretrain.pth",
    "encoder_dim": 768,
    "memory_dim": 128,
    "value_dim": 8,
    "geometry_dim": 6,
    "storage_dtype": "float16",
    "source": "labeled_only",
}

ENCODER_PC_STATIC_COMPAT_KEYS = tuple(ENCODER_PC_STATIC_COMPAT_META)
ENCODER_PC_REQUIRED_COMPAT_KEYS = (
    *ENCODER_PC_STATIC_COMPAT_KEYS,
    "dino_weight_fingerprint",
    "producer_fingerprint",
    "labeled_split_fingerprint",
)

_ROUTE_FLOAT_FIELDS = (
    "route_keys",
    "cls4_keys",
    "f4_global_keys",
    "f3_boundary_keys",
)
_PARENT_FLOAT_FIELDS = ("f3_parent_keys", "values", "geometry", "reliability")
_PARENT_INDEX_FIELDS = ("child_ptr", "image_index", "region_id", "flat_index")
_CHILD_FLOAT_FIELDS = ("f2_child_keys", "f1_detail_keys", "geometry")
_CHILD_INDEX_FIELDS = ("image_index", "flat_index")

_INDEX_DTYPES = {
    "child_ptr": torch.int32,
    "image_index": torch.int32,
    "region_id": torch.int16,
    "flat_index": torch.int16,
}

_SCHEMA_REBUILD_MESSAGE = (
    "Decoder-side PC-HBM memory is incompatible with encoder-side schema v3; "
    "rebuild memory from the labeled split."
)


@dataclass(frozen=True)
class EncoderMemoryCompatibilityResult:
    """Boolean-compatible compatibility result with a stable reason string."""

    ok: bool
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.ok

    def __iter__(self) -> Iterator[object]:
        yield self.ok
        yield self.reason


def build_encoder_memory_compat_meta(
    *,
    dino_weight_fingerprint: str,
    producer_fingerprint: str,
    labeled_split_fingerprint: str,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the complete, decoder-independent schema-v3 compatibility contract."""

    meta = dict(ENCODER_PC_STATIC_COMPAT_META)
    if overrides:
        meta.update(dict(overrides))
    meta["dino_weight_fingerprint"] = _required_fingerprint(
        dino_weight_fingerprint, "dino_weight_fingerprint"
    )
    meta["producer_fingerprint"] = _required_fingerprint(
        producer_fingerprint, "producer_fingerprint"
    )
    meta["labeled_split_fingerprint"] = _required_fingerprint(
        labeled_split_fingerprint, "labeled_split_fingerprint"
    )
    _validate_compat_meta(meta)
    return meta


class EncoderPCMemory:
    """Tensor-only encoder-side PC-HBM memory using schema v3.

    Floating tensors are always detached CPU FP16 tensors. Prototype metadata
    is stored in typed tensors; the route table is the only place where image
    identifiers are retained as Python strings.
    """

    DEFAULT_SCHEMA_VERSION = ENCODER_PC_MEMORY_SCHEMA_VERSION
    FORMAT_VERSION = ENCODER_PC_MEMORY_FORMAT_VERSION
    MEMORY_DIM = 128
    VALUE_DIM = 8
    GEOMETRY_DIM = 6

    def __init__(
        self,
        memory_dim: int = MEMORY_DIM,
        value_dim: int = VALUE_DIM,
        geometry_dim: int = GEOMETRY_DIM,
        *,
        storage_dtype: torch.dtype | str = torch.float16,
        compat_meta: Mapping[str, Any] | None = None,
    ) -> None:
        self.memory_dim = int(memory_dim)
        self.value_dim = int(value_dim)
        self.geometry_dim = int(geometry_dim)
        if (
            self.memory_dim,
            self.value_dim,
            self.geometry_dim,
        ) != (self.MEMORY_DIM, self.VALUE_DIM, self.GEOMETRY_DIM):
            raise ValueError("Encoder PC-HBM dimensions are fixed to 128/8/6")
        try:
            self.storage_dtype = _parse_storage_dtype(storage_dtype)
        except ValueError as error:
            raise ValueError(
                "Encoder PC-HBM memory storage dtype is fixed to float16"
            ) from error
        if self.storage_dtype != torch.float16:
            raise ValueError("Encoder PC-HBM memory storage dtype is fixed to float16")
        self._initial_compat_meta = dict(compat_meta or {})
        self.clear()

    def clear(self) -> None:
        self._route_lists: dict[str, list[torch.Tensor]] = {
            name: [] for name in _ROUTE_FLOAT_FIELDS
        }
        self._route_image_ids: list[str] = []
        self._parent_lists: dict[str, list[torch.Tensor]] = {
            name: [] for name in (*_PARENT_FLOAT_FIELDS, *_PARENT_INDEX_FIELDS)
        }
        self._child_lists: dict[str, list[torch.Tensor]] = {
            name: [] for name in (*_CHILD_FLOAT_FIELDS, *_CHILD_INDEX_FIELDS)
        }
        self.route: dict[str, Any] = {}
        self.parent: dict[str, torch.Tensor] = {}
        self.child: dict[str, torch.Tensor] = {}
        self.compat_meta: dict[str, Any] = dict(self._initial_compat_meta)
        self._finalized = False

    @property
    def num_images(self) -> int:
        if self._finalized:
            return len(self.route.get("image_ids", []))
        return len(self._route_image_ids)

    @property
    def num_parents(self) -> int:
        if self._finalized:
            value = self.parent.get("f3_parent_keys")
            return 0 if value is None else int(value.shape[0])
        return sum(int(item.shape[0]) for item in self._parent_lists["f3_parent_keys"])

    @property
    def num_children(self) -> int:
        if self._finalized:
            value = self.child.get("f2_child_keys")
            return 0 if value is None else int(value.shape[0])
        return sum(int(item.shape[0]) for item in self._child_lists["f2_child_keys"])

    def append(self, entries: Mapping[str, Any]) -> None:
        """Append one builder entry whose image and child indices are local.

        The entry must contain tensorized ``route``, ``parent`` and ``child``
        groups. Local image indices are offset by the current route-bank size;
        non-negative child pointers are offset by the current child-bank size.
        """

        self._ensure_mutable()
        source = str(entries.get("source", "labeled_only"))
        _validate_labeled_source(source)
        route = _require_mapping(entries, "route")
        parent = _require_mapping(entries, "parent")
        child = _require_mapping(entries, "child")
        route_offset = self.num_images
        child_offset = self.num_children

        self.append_route(
            **{name: route[name] for name in _ROUTE_FLOAT_FIELDS},
            image_ids=route["image_ids"],
            source=source,
        )
        child_image_index = _require_integer_vector(child["image_index"], "image_index")
        parent_image_index = _require_integer_vector(parent["image_index"], "image_index")
        child_ptr = _require_integer_vector(parent["child_ptr"], "child_ptr")
        if (child_image_index < 0).any() or (parent_image_index < 0).any():
            raise ValueError("Local image_index values must be non-negative")
        child_ptr = torch.where(child_ptr >= 0, child_ptr + child_offset, child_ptr)
        self.append_child(
            f2_child_keys=child["f2_child_keys"],
            f1_detail_keys=child["f1_detail_keys"],
            geometry=child["geometry"],
            image_index=child_image_index + route_offset,
            flat_index=child["flat_index"],
            source=source,
        )
        self.append_parent(
            f3_parent_keys=parent["f3_parent_keys"],
            values=parent["values"],
            geometry=parent["geometry"],
            child_ptr=child_ptr,
            image_index=parent_image_index + route_offset,
            region_id=parent["region_id"],
            flat_index=parent["flat_index"],
            reliability=parent["reliability"],
            source=source,
        )

    def append_route(
        self,
        *,
        route_keys: torch.Tensor,
        cls4_keys: torch.Tensor,
        f4_global_keys: torch.Tensor,
        f3_boundary_keys: torch.Tensor,
        image_ids: Sequence[object],
        source: str = "labeled_only",
    ) -> None:
        self._ensure_mutable()
        _validate_labeled_source(source)
        tensors = {
            "route_keys": _store_float_matrix(route_keys, self.memory_dim, "route_keys"),
            "cls4_keys": _store_float_matrix(cls4_keys, self.memory_dim, "cls4_keys"),
            "f4_global_keys": _store_float_matrix(
                f4_global_keys, self.memory_dim, "f4_global_keys"
            ),
            "f3_boundary_keys": _store_float_matrix(
                f3_boundary_keys, self.memory_dim, "f3_boundary_keys"
            ),
        }
        row_count = _common_row_count(tensors, "route")
        canonical_ids = [_canonical_image_id(item) for item in image_ids]
        if len(canonical_ids) != row_count:
            raise ValueError(
                f"route image_ids has {len(canonical_ids)} entries for {row_count} rows"
            )
        all_ids = [*self._route_image_ids, *canonical_ids]
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("Encoder PC-HBM route image_ids must be globally unique")
        for name, value in tensors.items():
            self._route_lists[name].append(value)
        self._route_image_ids.extend(canonical_ids)

    def append_parent(
        self,
        *,
        f3_parent_keys: torch.Tensor,
        values: torch.Tensor,
        geometry: torch.Tensor,
        child_ptr: torch.Tensor,
        image_index: torch.Tensor,
        region_id: torch.Tensor,
        flat_index: torch.Tensor,
        reliability: torch.Tensor,
        source: str = "labeled_only",
    ) -> None:
        self._ensure_mutable()
        _validate_labeled_source(source)
        tensors = {
            "f3_parent_keys": _store_float_matrix(
                f3_parent_keys, self.memory_dim, "f3_parent_keys"
            ),
            "values": _store_float_matrix(values, self.value_dim, "values"),
            "geometry": _store_float_matrix(geometry, self.geometry_dim, "geometry"),
            "reliability": _store_float_vector(reliability, "reliability"),
            "child_ptr": _store_index(child_ptr, "child_ptr"),
            "image_index": _store_index(image_index, "image_index"),
            "region_id": _store_index(region_id, "region_id"),
            "flat_index": _store_index(flat_index, "flat_index"),
        }
        _common_row_count(tensors, "parent")
        for name, value in tensors.items():
            self._parent_lists[name].append(value)

    def append_child(
        self,
        *,
        f2_child_keys: torch.Tensor,
        f1_detail_keys: torch.Tensor,
        geometry: torch.Tensor,
        image_index: torch.Tensor,
        flat_index: torch.Tensor,
        source: str = "labeled_only",
    ) -> torch.Tensor:
        self._ensure_mutable()
        _validate_labeled_source(source)
        tensors = {
            "f2_child_keys": _store_float_matrix(
                f2_child_keys, self.memory_dim, "f2_child_keys"
            ),
            "f1_detail_keys": _store_float_matrix(
                f1_detail_keys, self.memory_dim, "f1_detail_keys"
            ),
            "geometry": _store_float_matrix(geometry, self.geometry_dim, "geometry"),
            "image_index": _store_index(image_index, "image_index"),
            "flat_index": _store_index(flat_index, "flat_index"),
        }
        row_count = _common_row_count(tensors, "child")
        offset = self.num_children
        for name, value in tensors.items():
            self._child_lists[name].append(value)
        return torch.arange(offset, offset + row_count, dtype=torch.int32)

    def finalize(
        self,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float16,
        *,
        compat_meta: Mapping[str, Any] | None = None,
    ) -> None:
        self._ensure_mutable()
        if torch.device(device).type != "cpu" or dtype != torch.float16:
            raise ValueError("Encoder PC-HBM memory must be finalized as CPU float16")
        self.route = {
            name: _cat_or_empty(items, self.memory_dim, torch.float16)
            for name, items in self._route_lists.items()
        }
        self.route["image_ids"] = list(self._route_image_ids)
        self.parent = {
            "f3_parent_keys": _cat_or_empty(
                self._parent_lists["f3_parent_keys"], self.memory_dim, torch.float16
            ),
            "values": _cat_or_empty(
                self._parent_lists["values"], self.value_dim, torch.float16
            ),
            "geometry": _cat_or_empty(
                self._parent_lists["geometry"], self.geometry_dim, torch.float16
            ),
            "reliability": _cat_vector_or_empty(
                self._parent_lists["reliability"], torch.float16
            ),
            **{
                name: _cat_vector_or_empty(self._parent_lists[name], _INDEX_DTYPES[name])
                for name in _PARENT_INDEX_FIELDS
            },
        }
        self.child = {
            "f2_child_keys": _cat_or_empty(
                self._child_lists["f2_child_keys"], self.memory_dim, torch.float16
            ),
            "f1_detail_keys": _cat_or_empty(
                self._child_lists["f1_detail_keys"], self.memory_dim, torch.float16
            ),
            "geometry": _cat_or_empty(
                self._child_lists["geometry"], self.geometry_dim, torch.float16
            ),
            **{
                name: _cat_vector_or_empty(self._child_lists[name], _INDEX_DTYPES[name])
                for name in _CHILD_INDEX_FIELDS
            },
        }
        self.compat_meta = _merge_meta_strict(
            ENCODER_PC_STATIC_COMPAT_META,
            self._initial_compat_meta,
            dict(compat_meta or {}),
        )
        self._finalized = True
        try:
            self.validate()
        except Exception:
            self._finalized = False
            raise

    def is_ready(self) -> bool:
        return (
            self._finalized
            and self.num_images > 0
            and self.num_parents > 0
            and self.num_children > 0
        )

    def validate(self) -> None:
        """Raise on any schema, storage, shape, or tensorized-metadata violation."""

        if not self._finalized:
            raise RuntimeError("Encoder PC-HBM memory is not finalized")
        _validate_compat_meta(self.compat_meta)
        _validate_exact_fields(
            self.route, (*_ROUTE_FLOAT_FIELDS, "image_ids"), "route"
        )
        _validate_exact_fields(
            self.parent, (*_PARENT_FLOAT_FIELDS, *_PARENT_INDEX_FIELDS), "parent"
        )
        _validate_exact_fields(
            self.child, (*_CHILD_FLOAT_FIELDS, *_CHILD_INDEX_FIELDS), "child"
        )

        route_rows = _validate_float_group(
            self.route, {name: self.memory_dim for name in _ROUTE_FLOAT_FIELDS}, "route"
        )
        image_ids = self.route["image_ids"]
        if not isinstance(image_ids, list) or any(
            not isinstance(item, str) or not item for item in image_ids
        ):
            raise TypeError("route.image_ids must be a list of non-empty strings")
        if len(image_ids) != route_rows or len(set(image_ids)) != len(image_ids):
            raise ValueError("route.image_ids must be unique and match the route row count")

        parent_rows = _validate_float_group(
            self.parent,
            {
                "f3_parent_keys": self.memory_dim,
                "values": self.value_dim,
                "geometry": self.geometry_dim,
            },
            "parent",
        )
        reliability = self.parent["reliability"]
        _validate_float_vector(reliability, parent_rows, "parent.reliability")
        _validate_index_group(self.parent, _PARENT_INDEX_FIELDS, parent_rows, "parent")

        child_rows = _validate_float_group(
            self.child,
            {
                "f2_child_keys": self.memory_dim,
                "f1_detail_keys": self.memory_dim,
                "geometry": self.geometry_dim,
            },
            "child",
        )
        _validate_index_group(self.child, _CHILD_INDEX_FIELDS, child_rows, "child")
        if route_rows == 0 or parent_rows == 0 or child_rows == 0:
            raise ValueError("Encoder PC-HBM route, parent, and child banks must be non-empty")

        _validate_range(self.parent["image_index"], 0, route_rows, "parent.image_index")
        _validate_range(self.child["image_index"], 0, route_rows, "child.image_index")
        _validate_range(self.parent["region_id"], 0, 4, "parent.region_id")
        token_hw = tuple(int(item) for item in self.compat_meta["token_hw"])
        token_count = token_hw[0] * token_hw[1]
        _validate_range(self.parent["flat_index"], 0, token_count, "parent.flat_index")
        _validate_range(self.child["flat_index"], 0, token_count, "child.flat_index")
        child_ptr = self.parent["child_ptr"]
        if ((child_ptr < -1) | (child_ptr >= child_rows)).any():
            raise ValueError("parent.child_ptr must be -1 or reference a valid child row")
        if ((reliability < 0) | (reliability > 1)).any():
            raise ValueError("parent.reliability must be in [0, 1]")
        if not torch.allclose(
            self.parent["values"][:, 7], reliability, atol=1e-3, rtol=0.0
        ):
            raise ValueError("parent.values[:, 7] must equal parent.reliability")

    def validate_compat(
        self,
        expected: Mapping[str, Any] | object | None,
        *,
        require_producer_match: bool = True,
        require_split_match: bool = True,
    ) -> EncoderMemoryCompatibilityResult:
        if not self.is_ready():
            return EncoderMemoryCompatibilityResult(False, "memory_not_ready")
        try:
            self.validate()
        except (TypeError, ValueError, RuntimeError) as error:
            return EncoderMemoryCompatibilityResult(False, f"invalid_memory:{error}")
        if expected is None:
            return EncoderMemoryCompatibilityResult(True, None)
        if not isinstance(expected, Mapping):
            builder = getattr(expected, "expected_memory_meta", None)
            if not callable(builder):
                raise TypeError(
                    "expected compatibility data must be a mapping or encoder PC config"
                )
            expected = builder()
        expected_schema = int(expected.get("schema_version", 0))
        if expected_schema != self.DEFAULT_SCHEMA_VERSION:
            return EncoderMemoryCompatibilityResult(
                False, f"unsupported_expected_memory_schema:{expected_schema}"
            )
        keys = list(ENCODER_PC_STATIC_COMPAT_KEYS)
        if require_producer_match:
            keys.append("producer_fingerprint")
        if require_split_match:
            keys.append("labeled_split_fingerprint")
        for key in keys:
            if key not in expected:
                return EncoderMemoryCompatibilityResult(
                    False, f"missing_expected_compat_key:{key}"
                )
            if _canonical_meta(self.compat_meta[key]) != _canonical_meta(expected[key]):
                return EncoderMemoryCompatibilityResult(False, f"compat_mismatch:{key}")
        return EncoderMemoryCompatibilityResult(True, None)

    validate_compatibility = validate_compat

    def assert_compatible(
        self,
        expected: Mapping[str, Any] | object,
        *,
        require_producer_match: bool = True,
        require_split_match: bool = True,
    ) -> None:
        result = self.validate_compat(
            expected,
            require_producer_match=require_producer_match,
            require_split_match=require_split_match,
        )
        if not result:
            raise RuntimeError(
                f"Incompatible encoder PC-HBM memory ({result.reason}); "
                "rebuild memory from the labeled split."
            )

    def state_dict(self) -> dict[str, Any]:
        if not self._finalized:
            raise RuntimeError("Cannot serialize encoder PC-HBM memory before finalize()")
        self.validate()
        return {
            "format_version": self.FORMAT_VERSION,
            "schema_version": self.DEFAULT_SCHEMA_VERSION,
            "compat_meta": dict(self.compat_meta),
            "memory_dim": self.memory_dim,
            "value_dim": self.value_dim,
            "geometry_dim": self.geometry_dim,
            "storage_dtype": "float16",
            "route": _cpu_state_copy(self.route),
            "parent": _cpu_state_copy(self.parent),
            "child": _cpu_state_copy(self.child),
            "finalized": True,
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any] | None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Load schema v3 and explicitly reject decoder-side v1/v2 states."""

        self.clear()
        if not state:
            return
        if device is not None and torch.device(device).type != "cpu":
            raise ValueError("Loaded encoder PC-HBM memory must remain on CPU")
        if dtype is not None and dtype != torch.float16:
            raise ValueError("Loaded encoder PC-HBM memory must remain float16")
        outer = state
        if "memory" in state and isinstance(state["memory"], Mapping):
            state = state["memory"]
        outer_meta = dict(outer.get("compat_meta", {}) or {})
        inner_meta = dict(state.get("compat_meta", {}) or {})
        schema_values = {
            int(value)
            for value in (
                outer.get("schema_version"),
                state.get("schema_version"),
                outer_meta.get("schema_version"),
                inner_meta.get("schema_version"),
            )
            if value is not None
        }
        if len(schema_values) > 1:
            raise RuntimeError(
                f"Conflicting encoder PC-HBM memory schema declarations: {sorted(schema_values)}"
            )
        declared_schema = next(iter(schema_values), 1)
        if declared_schema in (1, 2):
            raise RuntimeError(_SCHEMA_REBUILD_MESSAGE)
        if declared_schema != self.DEFAULT_SCHEMA_VERSION:
            raise RuntimeError(
                f"Unsupported encoder PC-HBM memory schema v{declared_schema}; "
                "rebuild memory from the labeled split."
            )
        format_version = int(state.get("format_version", 0))
        if format_version != self.FORMAT_VERSION:
            raise RuntimeError(
                f"Unsupported encoder PC-HBM memory format v{format_version}; expected v3"
            )
        for key, expected in (
            ("memory_dim", self.memory_dim),
            ("value_dim", self.value_dim),
            ("geometry_dim", self.geometry_dim),
        ):
            actual = int(state.get(key, -1))
            if actual != expected:
                raise ValueError(
                    f"Encoder memory {key}={actual} is incompatible with expected {expected}"
                )
        if _parse_storage_dtype(state.get("storage_dtype", "")) != torch.float16:
            raise ValueError("Loaded encoder PC-HBM memory must remain float16")
        if not bool(state.get("finalized", False)):
            raise RuntimeError("Cannot load an unfinalized encoder PC-HBM memory state")

        raw_route = _require_mapping(state, "route")
        raw_parent = _require_mapping(state, "parent")
        raw_child = _require_mapping(state, "child")
        _validate_exact_fields(raw_route, (*_ROUTE_FLOAT_FIELDS, "image_ids"), "route")
        _validate_exact_fields(
            raw_parent, (*_PARENT_FLOAT_FIELDS, *_PARENT_INDEX_FIELDS), "parent"
        )
        _validate_exact_fields(
            raw_child, (*_CHILD_FLOAT_FIELDS, *_CHILD_INDEX_FIELDS), "child"
        )
        self.route = {
            name: _load_float_matrix(raw_route[name], self.memory_dim, f"route.{name}")
            for name in _ROUTE_FLOAT_FIELDS
        }
        raw_image_ids = raw_route["image_ids"]
        if not isinstance(raw_image_ids, Sequence) or isinstance(raw_image_ids, (str, bytes)):
            raise TypeError("route.image_ids must be a sequence of strings")
        self.route["image_ids"] = [_canonical_image_id(item) for item in raw_image_ids]
        self.parent = {
            "f3_parent_keys": _load_float_matrix(
                raw_parent["f3_parent_keys"], self.memory_dim, "parent.f3_parent_keys"
            ),
            "values": _load_float_matrix(
                raw_parent["values"], self.value_dim, "parent.values"
            ),
            "geometry": _load_float_matrix(
                raw_parent["geometry"], self.geometry_dim, "parent.geometry"
            ),
            "reliability": _load_float_vector(
                raw_parent["reliability"], "parent.reliability"
            ),
            **{
                name: _load_index(raw_parent[name], name, f"parent.{name}")
                for name in _PARENT_INDEX_FIELDS
            },
        }
        self.child = {
            "f2_child_keys": _load_float_matrix(
                raw_child["f2_child_keys"], self.memory_dim, "child.f2_child_keys"
            ),
            "f1_detail_keys": _load_float_matrix(
                raw_child["f1_detail_keys"], self.memory_dim, "child.f1_detail_keys"
            ),
            "geometry": _load_float_matrix(
                raw_child["geometry"], self.geometry_dim, "child.geometry"
            ),
            **{
                name: _load_index(raw_child[name], name, f"child.{name}")
                for name in _CHILD_INDEX_FIELDS
            },
        }
        self.compat_meta = _merge_meta_strict(
            ENCODER_PC_STATIC_COMPAT_META,
            outer_meta,
            inner_meta,
        )
        self._finalized = True
        try:
            self.validate()
        except Exception:
            self._finalized = False
            raise

    @classmethod
    def from_state_dict(
        cls,
        state: Mapping[str, Any],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "EncoderPCMemory":
        memory = cls()
        memory.load_state_dict(state, device=device, dtype=dtype)
        return memory

    def diagnostic_string(self) -> str:
        return (
            f"EncoderPCMemory(schema=3, ready={self.is_ready()}, images={self.num_images}, "
            f"parents={self.num_parents}, children={self.num_children}, "
            "device=cpu, dtype=float16)"
        )

    def _ensure_mutable(self) -> None:
        if self._finalized:
            raise RuntimeError("Encoder PC-HBM memory is finalized; call clear() before append")


def _validate_labeled_source(source: str) -> None:
    if source not in {"labeled", "labeled_only"}:
        raise ValueError("Encoder PC-HBM memory accepts labeled entries only")


def _required_fingerprint(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    result = value.strip()
    if not result:
        raise ValueError(f"{name} must be a non-empty string")
    return result


def _parse_storage_dtype(value: torch.dtype | str) -> torch.dtype:
    if value is torch.float16:
        return torch.float16
    normalized = str(value).lower().replace("torch.", "")
    if normalized in {"float16", "fp16", "half"}:
        return torch.float16
    raise ValueError(f"Unsupported encoder PC-HBM storage dtype: {value}")


def _store_float_matrix(value: torch.Tensor, width: int, name: str) -> torch.Tensor:
    _require_float_tensor(value, name)
    if value.ndim != 2 or int(value.shape[1]) != width:
        raise ValueError(f"{name} must have shape [N,{width}], got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value.detach().to(device="cpu", dtype=torch.float16).contiguous()


def _store_float_vector(value: torch.Tensor, name: str) -> torch.Tensor:
    _require_float_tensor(value, name)
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape [N], got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value.detach().to(device="cpu", dtype=torch.float16).contiguous()


def _store_index(value: torch.Tensor, name: str) -> torch.Tensor:
    value = _require_integer_vector(value, name)
    target_dtype = _INDEX_DTYPES[name]
    bounds = torch.iinfo(target_dtype)
    if value.numel() and ((value < bounds.min) | (value > bounds.max)).any():
        raise ValueError(f"{name} cannot be represented as {target_dtype}")
    return value.detach().to(device="cpu", dtype=target_dtype).contiguous()


def _require_float_tensor(value: object, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point tensor")


def _require_integer_vector(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be an integer tensor")
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape [N], got {tuple(value.shape)}")
    if value.is_floating_point() or value.is_complex() or value.dtype == torch.bool:
        raise TypeError(f"{name} must be an integer tensor")
    return value


def _common_row_count(tensors: Mapping[str, torch.Tensor], group: str) -> int:
    counts = {int(value.shape[0]) for value in tensors.values()}
    if len(counts) != 1:
        raise ValueError(f"{group} fields have inconsistent row counts: {sorted(counts)}")
    return next(iter(counts), 0)


def _cat_or_empty(
    items: Sequence[torch.Tensor], width: int, dtype: torch.dtype
) -> torch.Tensor:
    if not items:
        return torch.empty((0, width), dtype=dtype, device="cpu")
    return torch.cat(tuple(items), dim=0).to(device="cpu", dtype=dtype).contiguous()


def _cat_vector_or_empty(
    items: Sequence[torch.Tensor], dtype: torch.dtype
) -> torch.Tensor:
    if not items:
        return torch.empty((0,), dtype=dtype, device="cpu")
    return torch.cat(tuple(items), dim=0).to(device="cpu", dtype=dtype).contiguous()


def _validate_compat_meta(meta: Mapping[str, Any]) -> None:
    missing = [key for key in ENCODER_PC_REQUIRED_COMPAT_KEYS if key not in meta]
    if missing:
        raise ValueError(f"Missing encoder memory compatibility key: {missing[0]}")
    for key, expected in ENCODER_PC_STATIC_COMPAT_META.items():
        if _canonical_meta(meta[key]) != _canonical_meta(expected):
            raise ValueError(f"Invalid encoder memory compatibility value for {key}")
    _required_fingerprint(meta["producer_fingerprint"], "producer_fingerprint")
    _required_fingerprint(
        meta["dino_weight_fingerprint"], "dino_weight_fingerprint"
    )
    _required_fingerprint(
        meta["labeled_split_fingerprint"], "labeled_split_fingerprint"
    )


def _merge_meta_strict(*parts: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for part in parts:
        for key, value in part.items():
            if key in result and _canonical_meta(result[key]) != _canonical_meta(value):
                raise RuntimeError(f"Conflicting encoder memory metadata for {key}")
            result[key] = value
    return result


def _canonical_meta(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return tuple(_canonical_meta(item) for item in value)
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _canonical_meta(item)) for key, item in value.items()))
    return value


def _canonical_image_id(value: object) -> str:
    result = str(value).strip()
    if not result:
        raise ValueError("Encoder PC-HBM image IDs must be non-empty")
    return result


def _require_mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise TypeError(f"{key} must be a mapping")
    return result


def _validate_exact_fields(
    group: Mapping[str, Any], expected: Sequence[str], name: str
) -> None:
    actual = set(group)
    wanted = set(expected)
    missing = sorted(wanted - actual)
    unexpected = sorted(actual - wanted)
    if missing:
        raise ValueError(f"{name} is missing required field: {missing[0]}")
    if unexpected:
        raise ValueError(f"{name} contains unsupported field: {unexpected[0]}")


def _validate_cpu_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must remain on CPU")
    if value.requires_grad:
        raise ValueError(f"{name} must be detached")
    return value


def _validate_float_group(
    group: Mapping[str, Any], widths: Mapping[str, int], name: str
) -> int:
    counts: set[int] = set()
    for field, width in widths.items():
        value = _validate_cpu_tensor(group[field], f"{name}.{field}")
        if value.dtype != torch.float16:
            raise ValueError(f"{name}.{field} must be float16")
        if value.ndim != 2 or int(value.shape[1]) != width:
            raise ValueError(f"{name}.{field} must have shape [N,{width}]")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name}.{field} contains non-finite values")
        counts.add(int(value.shape[0]))
    if len(counts) != 1:
        raise ValueError(f"{name} floating fields have inconsistent row counts")
    return next(iter(counts), 0)


def _validate_float_vector(value: object, rows: int, name: str) -> None:
    value = _validate_cpu_tensor(value, name)
    if value.dtype != torch.float16 or value.ndim != 1 or int(value.shape[0]) != rows:
        raise ValueError(f"{name} must be float16 with shape [{rows}]")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")


def _validate_index_group(
    group: Mapping[str, Any], fields: Sequence[str], rows: int, name: str
) -> None:
    for field in fields:
        value = _validate_cpu_tensor(group[field], f"{name}.{field}")
        if value.dtype != _INDEX_DTYPES[field]:
            raise ValueError(f"{name}.{field} must use dtype {_INDEX_DTYPES[field]}")
        if value.ndim != 1 or int(value.shape[0]) != rows:
            raise ValueError(f"{name}.{field} must have shape [{rows}]")


def _validate_range(
    value: torch.Tensor, lower: int, upper_exclusive: int, name: str
) -> None:
    if ((value < lower) | (value >= upper_exclusive)).any():
        raise ValueError(f"{name} must be in [{lower}, {upper_exclusive})")


def _cpu_state_copy(group: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in group.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.detach().cpu().clone()
        elif key == "image_ids" and isinstance(value, list):
            result[key] = list(value)
        else:
            raise TypeError(f"Unsupported state value for {key}: {type(value).__name__}")
    return result


def _load_float_matrix(value: object, width: int, name: str) -> torch.Tensor:
    value = _validate_cpu_tensor(value, name)
    if value.dtype != torch.float16:
        raise ValueError(f"{name} must be float16")
    if value.ndim != 2 or int(value.shape[1]) != width:
        raise ValueError(f"{name} must have shape [N,{width}], got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value.detach().clone().contiguous()


def _load_float_vector(value: object, name: str) -> torch.Tensor:
    value = _validate_cpu_tensor(value, name)
    if value.dtype != torch.float16:
        raise ValueError(f"{name} must be float16")
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape [N], got {tuple(value.shape)}")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value.detach().clone().contiguous()


def _load_index(value: object, field: str, name: str) -> torch.Tensor:
    value = _validate_cpu_tensor(value, name)
    if value.dtype != _INDEX_DTYPES[field]:
        raise ValueError(f"{name} must use dtype {_INDEX_DTYPES[field]}")
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape [N], got {tuple(value.shape)}")
    return value.detach().clone().contiguous()


__all__ = [
    "ENCODER_PC_MEMORY_ARCHITECTURE",
    "ENCODER_PC_MEMORY_SCHEMA_VERSION",
    "ENCODER_PC_MEMORY_FORMAT_VERSION",
    "ENCODER_PC_STATIC_COMPAT_META",
    "ENCODER_PC_STATIC_COMPAT_KEYS",
    "ENCODER_PC_REQUIRED_COMPAT_KEYS",
    "EncoderMemoryCompatibilityResult",
    "EncoderPCMemory",
    "build_encoder_memory_compat_meta",
]
