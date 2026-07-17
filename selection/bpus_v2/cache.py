"""Schema-2 identity-bound score and prototype caches for BPUS-v2."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from selection.artifacts import atomic_torch_save

from .boundary_score import BPUS_V2_FORMULA_VERSION
from .prototype import (
    BPUS_V2_PROTOTYPE_VERSION,
    PROTOTYPE_DIM,
    PROTOTYPE_LEVEL,
)


CACHE_SCHEMA_VERSION = 2
METHOD = "BPUS-v2"
SCORE_PAYLOAD_TYPE = "bpus_v2_scores"
PROTOTYPE_PAYLOAD_TYPE = "bpus_v2_prototypes"
_IDENTITY_FIELDS = (
    "catalog_fingerprint",
    "image_fingerprint",
    "dino_fingerprint",
    "selector_fingerprint",
    "preprocessing_fingerprint",
)


def _keys(sample_keys: Sequence[str], name: str = "sample_keys") -> list[str]:
    keys = [str(key) for key in sample_keys]
    if any(not key for key in keys):
        raise ValueError(f"{name} cannot contain empty keys.")
    if len(keys) != len(set(keys)):
        raise ValueError(f"{name} must contain unique keys.")
    return keys


def _key_order_fingerprint(sample_keys: Sequence[str]) -> str:
    encoded = json.dumps(list(sample_keys), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _version_fingerprint(kind: str, version: str) -> str:
    if not isinstance(version, str) or not version:
        raise ValueError(f"{kind}_version must be a non-empty string.")
    encoded = json.dumps(
        {"kind": kind, "version": version}, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


FORMULA_FINGERPRINT = _version_fingerprint(
    "formula", BPUS_V2_FORMULA_VERSION
)
PROTOTYPE_FINGERPRINT = _version_fingerprint(
    "prototype", BPUS_V2_PROTOTYPE_VERSION
)


def _identity(**values: str) -> dict[str, str]:
    identity: dict[str, str] = {}
    for field in _IDENTITY_FIELDS:
        value = values.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{field} must be a non-empty string.")
        identity[field] = value
    return identity


def _vector(value: Any, length: int, name: str, *, boolean: bool = False) -> torch.Tensor:
    dtype = torch.bool if boolean else torch.float32
    if value is None:
        raise ValueError(f"Cache is missing required field {name!r}.")
    try:
        tensor = torch.as_tensor(value, dtype=dtype, device="cpu").reshape(-1).contiguous()
    except (TypeError, RuntimeError, ValueError) as error:
        raise ValueError(f"{name} cannot be converted to a tensor.") from error
    if tensor.numel() != length:
        raise ValueError(f"{name} must contain {length} entries.")
    if not boolean and not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite.")
    return tensor


def _expect(payload: Mapping[str, Any], name: str, expected: Any) -> None:
    if name not in payload:
        raise ValueError(f"Cache is missing required field {name!r}.")
    if payload[name] != expected:
        raise ValueError(
            f"Cache field {name!r} mismatch: expected {expected!r}, got {payload[name]!r}."
        )


def _base_payload(
    payload_type: str,
    sample_keys: Sequence[str],
    *,
    catalog_fingerprint: str,
    image_fingerprint: str,
    dino_fingerprint: str,
    selector_fingerprint: str,
    preprocessing_fingerprint: str,
    formula_version: str,
    prototype_version: str,
) -> dict[str, Any]:
    keys = _keys(sample_keys)
    identity = _identity(
        catalog_fingerprint=catalog_fingerprint,
        image_fingerprint=image_fingerprint,
        dino_fingerprint=dino_fingerprint,
        selector_fingerprint=selector_fingerprint,
        preprocessing_fingerprint=preprocessing_fingerprint,
    )
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "method": METHOD,
        "payload_type": payload_type,
        "sample_keys": keys,
        "key_order_fingerprint": _key_order_fingerprint(keys),
        **identity,
        "formula_version": formula_version,
        "formula_fingerprint": _version_fingerprint("formula", formula_version),
        "prototype_version": prototype_version,
        "prototype_fingerprint": _version_fingerprint(
            "prototype", prototype_version
        ),
    }


def build_score_payload(
    sample_keys: Sequence[str],
    boundary_disagreement,
    global_disagreement,
    boundary_value,
    boundary_mass,
    valid_boundary,
    *,
    catalog_fingerprint: str,
    image_fingerprint: str,
    dino_fingerprint: str,
    selector_fingerprint: str,
    preprocessing_fingerprint: str,
    boundary_mass_eps: float = 1e-6,
    formula_version: str = BPUS_V2_FORMULA_VERSION,
    prototype_version: str = BPUS_V2_PROTOTYPE_VERSION,
) -> dict[str, Any]:
    """Build a strict CPU score payload with complete dependency identity."""

    if not math.isfinite(boundary_mass_eps) or boundary_mass_eps <= 0.0:
        raise ValueError("boundary_mass_eps must be finite and positive.")
    payload = _base_payload(
        SCORE_PAYLOAD_TYPE,
        sample_keys,
        catalog_fingerprint=catalog_fingerprint,
        image_fingerprint=image_fingerprint,
        dino_fingerprint=dino_fingerprint,
        selector_fingerprint=selector_fingerprint,
        preprocessing_fingerprint=preprocessing_fingerprint,
        formula_version=formula_version,
        prototype_version=prototype_version,
    )
    length = len(payload["sample_keys"])
    payload.update(
        {
            "boundary_mass_eps": float(boundary_mass_eps),
            "boundary_disagreement": _vector(
                boundary_disagreement, length, "boundary_disagreement"
            ),
            "global_disagreement": _vector(
                global_disagreement, length, "global_disagreement"
            ),
            "boundary_value": _vector(boundary_value, length, "boundary_value"),
            "boundary_mass": _vector(boundary_mass, length, "boundary_mass"),
            "valid_boundary": _vector(
                valid_boundary, length, "valid_boundary", boolean=True
            ),
        }
    )
    _validate_score_data(payload)
    return payload


def _validate_base(
    payload: Mapping[str, Any],
    payload_type: str,
    *,
    expected_sample_keys: Sequence[str],
    expected_catalog_fingerprint: str,
    expected_image_fingerprint: str,
    expected_dino_fingerprint: str,
    expected_selector_fingerprint: str,
    expected_preprocessing_fingerprint: str,
    expected_formula_version: str,
    expected_prototype_version: str,
) -> list[str]:
    if not isinstance(payload, Mapping):
        raise TypeError("Cache payload must be a mapping.")
    keys = _keys(expected_sample_keys, "expected_sample_keys")
    _expect(payload, "schema_version", CACHE_SCHEMA_VERSION)
    _expect(payload, "method", METHOD)
    _expect(payload, "payload_type", payload_type)
    _expect(payload, "sample_keys", keys)
    _expect(payload, "key_order_fingerprint", _key_order_fingerprint(keys))
    expected_identity = _identity(
        catalog_fingerprint=expected_catalog_fingerprint,
        image_fingerprint=expected_image_fingerprint,
        dino_fingerprint=expected_dino_fingerprint,
        selector_fingerprint=expected_selector_fingerprint,
        preprocessing_fingerprint=expected_preprocessing_fingerprint,
    )
    for name, expected in expected_identity.items():
        _expect(payload, name, expected)
    _expect(payload, "formula_version", expected_formula_version)
    _expect(
        payload,
        "formula_fingerprint",
        _version_fingerprint("formula", expected_formula_version),
    )
    _expect(payload, "prototype_version", expected_prototype_version)
    _expect(
        payload,
        "prototype_fingerprint",
        _version_fingerprint("prototype", expected_prototype_version),
    )
    return keys


def _validate_score_data(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    length = len(payload["sample_keys"])
    result = {
        "boundary_disagreement": _vector(
            payload.get("boundary_disagreement"), length, "boundary_disagreement"
        ),
        "global_disagreement": _vector(
            payload.get("global_disagreement"), length, "global_disagreement"
        ),
        "boundary_value": _vector(
            payload.get("boundary_value"), length, "boundary_value"
        ),
        "boundary_mass": _vector(
            payload.get("boundary_mass"), length, "boundary_mass"
        ),
        "valid_boundary": _vector(
            payload.get("valid_boundary"), length, "valid_boundary", boolean=True
        ),
    }
    for name in (
        "boundary_disagreement",
        "global_disagreement",
        "boundary_value",
    ):
        if bool(((result[name] < 0.0) | (result[name] > 1.0)).any()):
            raise ValueError(f"{name} must lie in [0,1].")
    if bool((result["boundary_mass"] < 0.0).any()):
        raise ValueError("boundary_mass must be non-negative.")
    boundary_mass_eps = payload.get("boundary_mass_eps")
    if (
        not isinstance(boundary_mass_eps, (float, int))
        or not math.isfinite(float(boundary_mass_eps))
        or float(boundary_mass_eps) <= 0.0
    ):
        raise ValueError("Cache boundary_mass_eps must be finite and positive.")
    expected_valid = result["boundary_mass"] > float(boundary_mass_eps)
    if not torch.equal(result["valid_boundary"], expected_valid):
        raise ValueError("valid_boundary does not match boundary_mass thresholding.")
    if bool((result["boundary_value"][~result["valid_boundary"]] != 0.0).any()):
        raise ValueError("Invalid-boundary values must be exactly zero.")
    return result


def validate_score_payload(
    payload: Mapping[str, Any],
    *,
    expected_sample_keys: Sequence[str],
    expected_catalog_fingerprint: str,
    expected_image_fingerprint: str,
    expected_dino_fingerprint: str,
    expected_selector_fingerprint: str,
    expected_preprocessing_fingerprint: str,
    expected_boundary_mass_eps: float = 1e-6,
    expected_formula_version: str = BPUS_V2_FORMULA_VERSION,
    expected_prototype_version: str = BPUS_V2_PROTOTYPE_VERSION,
) -> dict[str, torch.Tensor]:
    """Validate score identity, schema, shapes, ranges, and boundary validity."""

    _validate_base(
        payload,
        SCORE_PAYLOAD_TYPE,
        expected_sample_keys=expected_sample_keys,
        expected_catalog_fingerprint=expected_catalog_fingerprint,
        expected_image_fingerprint=expected_image_fingerprint,
        expected_dino_fingerprint=expected_dino_fingerprint,
        expected_selector_fingerprint=expected_selector_fingerprint,
        expected_preprocessing_fingerprint=expected_preprocessing_fingerprint,
        expected_formula_version=expected_formula_version,
        expected_prototype_version=expected_prototype_version,
    )
    _expect(payload, "boundary_mass_eps", float(expected_boundary_mass_eps))
    return _validate_score_data(payload)


def build_prototype_payload(
    sample_keys: Sequence[str],
    prototypes: torch.Tensor,
    valid_boundary,
    *,
    catalog_fingerprint: str,
    image_fingerprint: str,
    dino_fingerprint: str,
    selector_fingerprint: str,
    preprocessing_fingerprint: str,
    formula_version: str = BPUS_V2_FORMULA_VERSION,
    prototype_version: str = BPUS_V2_PROTOTYPE_VERSION,
) -> dict[str, Any]:
    """Build a strict CPU float32 P2 prototype payload."""

    payload = _base_payload(
        PROTOTYPE_PAYLOAD_TYPE,
        sample_keys,
        catalog_fingerprint=catalog_fingerprint,
        image_fingerprint=image_fingerprint,
        dino_fingerprint=dino_fingerprint,
        selector_fingerprint=selector_fingerprint,
        preprocessing_fingerprint=preprocessing_fingerprint,
        formula_version=formula_version,
        prototype_version=prototype_version,
    )
    length = len(payload["sample_keys"])
    if not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 2:
        raise ValueError("prototypes must have shape [N,128].")
    tensor = prototypes.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if tuple(tensor.shape) != (length, PROTOTYPE_DIM):
        raise ValueError(
            f"prototypes must have shape [{length},{PROTOTYPE_DIM}], "
            f"found {tuple(tensor.shape)}."
        )
    valid = _vector(valid_boundary, length, "valid_boundary", boolean=True)
    payload.update(
        {
            "prototype_level": PROTOTYPE_LEVEL,
            "prototype_dim": PROTOTYPE_DIM,
            "prototypes": tensor,
            "valid_boundary": valid,
        }
    )
    _validate_prototype_data(payload)
    return payload


def _validate_prototype_data(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    length = len(payload["sample_keys"])
    _expect(payload, "prototype_level", PROTOTYPE_LEVEL)
    _expect(payload, "prototype_dim", PROTOTYPE_DIM)
    prototypes = payload.get("prototypes")
    if not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 2:
        raise ValueError("Cache prototypes must have shape [N,128].")
    prototypes = prototypes.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if tuple(prototypes.shape) != (length, PROTOTYPE_DIM):
        raise ValueError("Cache prototype shape does not match schema-2 metadata.")
    if not torch.isfinite(prototypes).all():
        raise ValueError("Cache prototypes must be finite.")
    valid = _vector(payload.get("valid_boundary"), length, "valid_boundary", boolean=True)
    if bool((prototypes[~valid] != 0.0).any()):
        raise ValueError("Invalid-boundary prototypes must be exactly zero.")
    norms = torch.linalg.vector_norm(prototypes[valid], dim=1)
    nonzero = norms > 0.0
    if bool(nonzero.any()) and not torch.allclose(
        norms[nonzero], torch.ones_like(norms[nonzero]), atol=1e-5, rtol=1e-5
    ):
        raise ValueError("Non-zero valid prototypes must be L2 normalized.")
    return {"prototypes": prototypes, "valid_boundary": valid}


def validate_prototype_payload(
    payload: Mapping[str, Any],
    *,
    expected_sample_keys: Sequence[str],
    expected_catalog_fingerprint: str,
    expected_image_fingerprint: str,
    expected_dino_fingerprint: str,
    expected_selector_fingerprint: str,
    expected_preprocessing_fingerprint: str,
    expected_prototype_dim: int = PROTOTYPE_DIM,
    expected_valid_boundary=None,
    expected_formula_version: str = BPUS_V2_FORMULA_VERSION,
    expected_prototype_version: str = BPUS_V2_PROTOTYPE_VERSION,
) -> dict[str, torch.Tensor]:
    """Validate prototype identity, P2 metadata, normalization, and validity."""

    _validate_base(
        payload,
        PROTOTYPE_PAYLOAD_TYPE,
        expected_sample_keys=expected_sample_keys,
        expected_catalog_fingerprint=expected_catalog_fingerprint,
        expected_image_fingerprint=expected_image_fingerprint,
        expected_dino_fingerprint=expected_dino_fingerprint,
        expected_selector_fingerprint=expected_selector_fingerprint,
        expected_preprocessing_fingerprint=expected_preprocessing_fingerprint,
        expected_formula_version=expected_formula_version,
        expected_prototype_version=expected_prototype_version,
    )
    if int(expected_prototype_dim) != PROTOTYPE_DIM:
        raise ValueError(f"BPUS-v2 prototype_dim must be {PROTOTYPE_DIM}.")
    result = _validate_prototype_data(payload)
    if expected_valid_boundary is not None:
        expected_valid = _vector(
            expected_valid_boundary,
            len(payload["sample_keys"]),
            "expected_valid_boundary",
            boolean=True,
        )
        if not torch.equal(result["valid_boundary"], expected_valid):
            raise ValueError("Prototype valid_boundary does not match the score cache.")
    return result


def save_cache_payload(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    refuse_mismatch: bool = True,
) -> None:
    """Atomically save a built schema-2 payload."""

    if not isinstance(payload, Mapping):
        raise TypeError("Cache payload must be a mapping.")
    _expect(payload, "schema_version", CACHE_SCHEMA_VERSION)
    if payload.get("payload_type") not in (SCORE_PAYLOAD_TYPE, PROTOTYPE_PAYLOAD_TYPE):
        raise ValueError("Unknown BPUS-v2 payload_type.")
    atomic_torch_save(dict(payload), path, refuse_mismatch=refuse_mismatch)


def load_cache_payload(path: str | Path) -> Mapping[str, Any]:
    """Load a cache payload without weakening subsequent identity validation."""

    try:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(Path(path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise TypeError("Cache payload must be a mapping.")
    return payload


build_score_cache = build_score_payload
validate_score_cache = validate_score_payload
build_prototype_cache = build_prototype_payload
validate_prototype_cache = validate_prototype_payload
