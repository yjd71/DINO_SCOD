"""Strict identity-bound payloads for PC-BPUS score and prototype caches."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .boundary_score import BOUNDARY_SCORE_VERSION
from .prototype import PROTOTYPE_VERSION


CACHE_SCHEMA_VERSION = 1
SCORE_PAYLOAD_TYPE = "pc_bpus_scores"
PROTOTYPE_PAYLOAD_TYPE = "pc_bpus_prototypes"
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
        "payload_type": payload_type,
        "sample_keys": keys,
        "key_order_fingerprint": _key_order_fingerprint(keys),
        **identity,
    }


def build_score_payload(
    sample_keys: Sequence[str],
    d_boundary,
    d_global,
    value,
    boundary_mass,
    valid_boundary,
    *,
    catalog_fingerprint: str,
    image_fingerprint: str,
    dino_fingerprint: str,
    selector_fingerprint: str,
    preprocessing_fingerprint: str,
    boundary_mass_eps: float = 1e-6,
    formula_version: str = BOUNDARY_SCORE_VERSION,
) -> dict[str, Any]:
    """Build a CPU payload whose identity covers every scoring dependency."""

    if not math.isfinite(boundary_mass_eps) or boundary_mass_eps <= 0.0:
        raise ValueError("boundary_mass_eps must be finite and positive.")
    if not isinstance(formula_version, str) or not formula_version:
        raise ValueError("formula_version must be a non-empty string.")
    payload = _base_payload(
        SCORE_PAYLOAD_TYPE,
        sample_keys,
        catalog_fingerprint=catalog_fingerprint,
        image_fingerprint=image_fingerprint,
        dino_fingerprint=dino_fingerprint,
        selector_fingerprint=selector_fingerprint,
        preprocessing_fingerprint=preprocessing_fingerprint,
    )
    length = len(payload["sample_keys"])
    payload.update(
        {
            "formula_version": formula_version,
            "boundary_mass_eps": float(boundary_mass_eps),
            "d_boundary": _vector(d_boundary, length, "d_boundary"),
            "d_global": _vector(d_global, length, "d_global"),
            "value": _vector(value, length, "value"),
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
) -> list[str]:
    if not isinstance(payload, Mapping):
        raise TypeError("Cache payload must be a mapping.")
    keys = _keys(expected_sample_keys, "expected_sample_keys")
    _expect(payload, "schema_version", CACHE_SCHEMA_VERSION)
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
    return keys


def _validate_score_data(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    length = len(payload["sample_keys"])
    result = {
        "d_boundary": _vector(payload.get("d_boundary"), length, "d_boundary"),
        "d_global": _vector(payload.get("d_global"), length, "d_global"),
        "value": _vector(payload.get("value"), length, "value"),
        "boundary_mass": _vector(payload.get("boundary_mass"), length, "boundary_mass"),
        "valid_boundary": _vector(
            payload.get("valid_boundary"), length, "valid_boundary", boolean=True
        ),
    }
    for name in ("d_boundary", "d_global", "value"):
        if bool(((result[name] < 0.0) | (result[name] > 1.0)).any()):
            raise ValueError(f"{name} must lie in [0,1].")
    if bool((result["boundary_mass"] < 0.0).any()):
        raise ValueError("boundary_mass must be non-negative.")
    boundary_mass_eps = payload.get("boundary_mass_eps")
    if not isinstance(boundary_mass_eps, (float, int)) or not math.isfinite(
        float(boundary_mass_eps)
    ) or float(boundary_mass_eps) <= 0.0:
        raise ValueError("Cache boundary_mass_eps must be finite and positive.")
    expected_valid = result["boundary_mass"] > float(boundary_mass_eps)
    if not torch.equal(result["valid_boundary"], expected_valid):
        raise ValueError("valid_boundary does not match boundary_mass thresholding.")
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
    expected_formula_version: str = BOUNDARY_SCORE_VERSION,
) -> dict[str, torch.Tensor]:
    """Validate score payload identity, key order, shapes and invariants."""

    _validate_base(
        payload,
        SCORE_PAYLOAD_TYPE,
        expected_sample_keys=expected_sample_keys,
        expected_catalog_fingerprint=expected_catalog_fingerprint,
        expected_image_fingerprint=expected_image_fingerprint,
        expected_dino_fingerprint=expected_dino_fingerprint,
        expected_selector_fingerprint=expected_selector_fingerprint,
        expected_preprocessing_fingerprint=expected_preprocessing_fingerprint,
    )
    _expect(payload, "formula_version", expected_formula_version)
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
    prototype_version: str = PROTOTYPE_VERSION,
) -> dict[str, Any]:
    """Build a strict CPU float32 P2 prototype payload."""

    if not isinstance(prototype_version, str) or not prototype_version:
        raise ValueError("prototype_version must be a non-empty string.")
    payload = _base_payload(
        PROTOTYPE_PAYLOAD_TYPE,
        sample_keys,
        catalog_fingerprint=catalog_fingerprint,
        image_fingerprint=image_fingerprint,
        dino_fingerprint=dino_fingerprint,
        selector_fingerprint=selector_fingerprint,
        preprocessing_fingerprint=preprocessing_fingerprint,
    )
    length = len(payload["sample_keys"])
    if not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 2:
        raise ValueError("prototypes must have shape [N,D].")
    tensor = prototypes.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if tensor.shape[0] != length or tensor.shape[1] <= 0:
        raise ValueError("prototypes shape must match sample_keys and have D>0.")
    valid = _vector(valid_boundary, length, "valid_boundary", boolean=True)
    payload.update(
        {
            "prototype_version": prototype_version,
            "feature_dim": int(tensor.shape[1]),
            "prototypes": tensor,
            "valid_boundary": valid,
        }
    )
    _validate_prototype_data(payload)
    return payload


def _validate_prototype_data(payload: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    length = len(payload["sample_keys"])
    feature_dim = payload.get("feature_dim")
    if not isinstance(feature_dim, int) or feature_dim <= 0:
        raise ValueError("feature_dim must be a positive integer.")
    prototypes = payload.get("prototypes")
    if not isinstance(prototypes, torch.Tensor) or prototypes.ndim != 2:
        raise ValueError("Cache prototypes must have shape [N,D].")
    prototypes = prototypes.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if tuple(prototypes.shape) != (length, feature_dim):
        raise ValueError("Cache prototype shape does not match metadata.")
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
    expected_feature_dim: int | None = None,
    expected_valid_boundary=None,
    expected_prototype_version: str = PROTOTYPE_VERSION,
) -> dict[str, torch.Tensor]:
    """Validate prototype payload identity, key order, shape and normalization."""

    _validate_base(
        payload,
        PROTOTYPE_PAYLOAD_TYPE,
        expected_sample_keys=expected_sample_keys,
        expected_catalog_fingerprint=expected_catalog_fingerprint,
        expected_image_fingerprint=expected_image_fingerprint,
        expected_dino_fingerprint=expected_dino_fingerprint,
        expected_selector_fingerprint=expected_selector_fingerprint,
        expected_preprocessing_fingerprint=expected_preprocessing_fingerprint,
    )
    _expect(payload, "prototype_version", expected_prototype_version)
    if expected_feature_dim is not None:
        _expect(payload, "feature_dim", int(expected_feature_dim))
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


# Cache-oriented aliases keep call sites readable while preserving the payload
# terminology used by formal artifact manifests.
build_score_cache = build_score_payload
validate_score_cache = validate_score_payload
build_prototype_cache = build_prototype_payload
validate_prototype_cache = validate_prototype_payload
