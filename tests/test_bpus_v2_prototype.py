from __future__ import annotations

import copy

import pytest
import torch

from selection.bpus_v2 import (
    BPUS_V2_FORMULA_VERSION,
    BPUS_V2_PROTOTYPE_VERSION,
    build_bpus_v2_prototype,
    build_prototype_payload,
    build_score_payload,
    load_cache_payload,
    save_cache_payload,
    validate_prototype_payload,
    validate_score_payload,
)


IDENTITY = {
    "catalog_fingerprint": "catalog",
    "image_fingerprint": "images",
    "dino_fingerprint": "encoder",
    "selector_fingerprint": "selector",
    "preprocessing_fingerprint": "preprocess",
}


def _expected_identity(keys: list[str]) -> dict[str, object]:
    return {
        "expected_sample_keys": keys,
        "expected_catalog_fingerprint": "catalog",
        "expected_image_fingerprint": "images",
        "expected_dino_fingerprint": "encoder",
        "expected_selector_fingerprint": "selector",
        "expected_preprocessing_fingerprint": "preprocess",
    }


def test_prototype_normalizes_each_location_and_final_aggregate() -> None:
    p2 = torch.zeros(1, 128, 28, 28)
    p2[0, 0, 4, 6] = 7.0
    p2[0, 1, 20, 22] = 3.0
    weights = torch.zeros(1, 1, 56, 56)
    weights[..., 8:10, 12:14] = 1.0

    prototype = build_bpus_v2_prototype(
        p2, weights, valid_boundary=torch.tensor([True])
    )

    assert prototype.shape == (1, 128)
    assert torch.linalg.vector_norm(prototype, dim=1).item() == pytest.approx(1.0)
    assert prototype[0, 0].item() > prototype[0, 1].item()


def test_invalid_boundary_prototype_is_exact_zero() -> None:
    prototype = build_bpus_v2_prototype(
        torch.randn(2, 128, 28, 28),
        torch.ones(2, 1, 56, 56),
        valid_boundary=torch.tensor([False, False]),
    )
    assert torch.count_nonzero(prototype).item() == 0


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("target", ["p2", "boundary_weight"])
def test_prototype_rejects_nonfinite_inputs(target: str, bad_value: float) -> None:
    p2 = torch.randn(1, 128, 28, 28)
    boundary_weight = torch.ones(1, 1, 56, 56)
    tensor = p2 if target == "p2" else boundary_weight
    tensor[0, 0, 0, 0] = bad_value

    with pytest.raises(ValueError, match=target):
        build_bpus_v2_prototype(p2, boundary_weight)


def test_prototype_rejects_nonfinite_intermediate_norm() -> None:
    p2 = torch.full(
        (1, 128, 28, 28), torch.finfo(torch.float32).max, dtype=torch.float32
    )

    with pytest.raises(ValueError, match="P2 point norm"):
        build_bpus_v2_prototype(p2, torch.ones(1, 1, 56, 56))


def test_prototype_rejects_nonfinite_intermediate_boundary_mass() -> None:
    boundary_weight = torch.full(
        (1, 1, 56, 56), torch.finfo(torch.float32).max, dtype=torch.float32
    )

    with pytest.raises(ValueError, match="boundary_weight mass"):
        build_bpus_v2_prototype(
            torch.randn(1, 128, 28, 28), boundary_weight
        )


def test_schema2_score_cache_round_trip_and_schema1_rejection(tmp_path) -> None:
    keys = ["TR-CAMO/a", "TR-COD10K/b"]
    payload = build_score_payload(
        keys,
        [0.4, 0.0],
        [0.2, 0.3],
        [0.32, 0.0],
        [1.0, 0.0],
        [True, False],
        **IDENTITY,
    )
    assert payload["schema_version"] == 2
    assert payload["formula_version"] == BPUS_V2_FORMULA_VERSION
    assert payload["prototype_version"] == BPUS_V2_PROTOTYPE_VERSION
    assert "formula_fingerprint" in payload
    assert "prototype_fingerprint" in payload
    assert "d_boundary" not in payload

    path = tmp_path / "pool_scores.pt"
    save_cache_payload(path, payload)
    loaded = load_cache_payload(path)
    result = validate_score_payload(loaded, **_expected_identity(keys))
    assert torch.equal(result["boundary_value"], torch.tensor([0.32, 0.0]))

    legacy = copy.deepcopy(payload)
    legacy["schema_version"] = 1
    with pytest.raises(ValueError, match="schema_version"):
        validate_score_payload(legacy, **_expected_identity(keys))

    tampered = copy.deepcopy(payload)
    tampered["selector_fingerprint"] = "other"
    with pytest.raises(ValueError, match="selector_fingerprint"):
        validate_score_payload(tampered, **_expected_identity(keys))


def test_schema2_prototype_cache_metadata_and_invalid_row_check() -> None:
    keys = ["a", "b"]
    prototypes = torch.zeros(2, 128)
    prototypes[0, 3] = 1.0
    payload = build_prototype_payload(
        keys, prototypes, [True, False], **IDENTITY
    )
    assert payload["prototype_level"] == "p2"
    assert payload["prototype_dim"] == 128
    result = validate_prototype_payload(
        payload,
        expected_valid_boundary=[True, False],
        **_expected_identity(keys),
    )
    assert torch.equal(result["prototypes"], prototypes)

    tampered = copy.deepcopy(payload)
    tampered["prototypes"][1, 0] = 1.0
    with pytest.raises(ValueError, match="exactly zero"):
        validate_prototype_payload(tampered, **_expected_identity(keys))
