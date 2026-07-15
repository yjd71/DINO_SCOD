from __future__ import annotations

import copy
import math

import pytest
import torch

from selection.pc_bpus import (
    build_boundary_prototype,
    build_prototype_payload,
    build_score_payload,
    validate_prototype_payload,
    validate_score_payload,
)


IDENTITY = {
    "catalog_fingerprint": "catalog",
    "image_fingerprint": "images",
    "dino_fingerprint": "dino",
    "selector_fingerprint": "selector",
    "preprocessing_fingerprint": "preprocess",
}


def test_prototype_normalizes_each_location_before_boundary_pooling() -> None:
    p2 = torch.tensor([[[[3.0, 0.0]], [[0.0, 4.0]]]])
    weights = torch.ones(1, 1, 1, 2)
    prototype = build_boundary_prototype(
        p2, weights, valid_boundary=torch.tensor([True])
    )

    expected = torch.tensor([[1.0 / math.sqrt(2.0), 1.0 / math.sqrt(2.0)]])
    assert torch.allclose(prototype, expected, atol=1e-6, rtol=0.0)
    assert torch.allclose(
        torch.linalg.vector_norm(prototype, dim=1), torch.ones(1), atol=1e-6
    )


def test_invalid_boundary_prototype_is_exactly_zero() -> None:
    p2 = torch.randn(2, 4, 3, 3)
    weights = torch.zeros(2, 1, 7, 7)
    weights[1, :, 2:5, 2:5] = 1.0
    prototype = build_boundary_prototype(p2, weights)

    assert torch.count_nonzero(prototype[0]) == 0
    assert torch.linalg.vector_norm(prototype[1], dim=0).item() == pytest.approx(
        1.0, abs=1e-6
    )


def test_score_payload_round_trip_checks_identity_order_and_validity() -> None:
    keys = ["TR-CAMO/a", "TR-COD10K/b"]
    payload = build_score_payload(
        keys,
        d_boundary=torch.tensor([0.0, 0.4]),
        d_global=torch.tensor([0.0, 0.1]),
        value=torch.tensor([0.0, 0.27]),
        boundary_mass=torch.tensor([0.0, 1e-3]),
        valid_boundary=torch.tensor([False, True]),
        **IDENTITY,
    )
    restored = validate_score_payload(
        payload,
        expected_sample_keys=keys,
        **{f"expected_{name}": value for name, value in IDENTITY.items()},
    )
    assert restored["value"].dtype == torch.float32
    assert torch.equal(restored["valid_boundary"], torch.tensor([False, True]))

    mismatch = copy.deepcopy(payload)
    mismatch["selector_fingerprint"] = "different"
    with pytest.raises(ValueError, match="selector_fingerprint"):
        validate_score_payload(
            mismatch,
            expected_sample_keys=keys,
            **{f"expected_{name}": value for name, value in IDENTITY.items()},
        )
    with pytest.raises(ValueError, match="sample_keys"):
        validate_score_payload(
            payload,
            expected_sample_keys=list(reversed(keys)),
            **{f"expected_{name}": value for name, value in IDENTITY.items()},
        )


def test_prototype_payload_round_trip_rejects_nonzero_invalid_row() -> None:
    keys = ["a", "b"]
    prototypes = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    payload = build_prototype_payload(
        keys,
        prototypes,
        valid_boundary=torch.tensor([True, False]),
        **IDENTITY,
    )
    restored = validate_prototype_payload(
        payload,
        expected_sample_keys=keys,
        expected_feature_dim=2,
        **{f"expected_{name}": value for name, value in IDENTITY.items()},
    )
    assert torch.equal(restored["prototypes"], prototypes)

    malformed = copy.deepcopy(payload)
    malformed["prototypes"][1, 0] = 1.0
    with pytest.raises(ValueError, match="exactly zero"):
        validate_prototype_payload(
            malformed,
            expected_sample_keys=keys,
            expected_feature_dim=2,
            **{f"expected_{name}": value for name, value in IDENTITY.items()},
        )
