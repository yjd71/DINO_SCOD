from __future__ import annotations

import json

import pytest
import torch

from select_pc_bacs import (
    _expected_seed_count,
    _validate_seed_for_mode,
    parse_args as parse_pc_bacs_args,
)
from selection.artifacts import (
    atomic_json_save,
    compute_catalog_fingerprint,
    compute_image_fingerprint,
    compute_key_order_fingerprint,
    load_split_keys,
    save_split_keys,
    stable_fingerprint,
)
from selection.protocol import SamplingProtocol


@pytest.mark.parametrize(
    ("counts", "name", "bootstrap"),
    [
        ((40, 200, 400), "kmeans_0040_0200_0400", 40),
        ((41, 202, 404), "scout_0041_0202_0404", 41),
    ],
)
def test_formal_protocols_are_named_and_validated(counts, name, bootstrap) -> None:
    protocol = SamplingProtocol.from_counts(counts)
    assert protocol.name == name
    assert protocol.target_counts == counts
    assert protocol.bootstrap_count == bootstrap
    assert protocol.is_formal is True


@pytest.mark.parametrize(
    "counts",
    [
        (40, 202, 404),
        (41, 200, 400),
        (41, 202),
        (41, 41, 404),
        (202, 41, 404),
        (0, 202, 404),
    ],
)
def test_nonformal_or_invalid_protocols_are_rejected_by_default(counts) -> None:
    with pytest.raises((TypeError, ValueError)):
        SamplingProtocol.from_counts(counts)


def test_custom_protocol_requires_explicit_debug_permission() -> None:
    custom = SamplingProtocol.from_counts((2, 4, 6), allow_custom=True)
    assert custom.name == "custom_0002_0004_0006"
    assert custom.bootstrap_count == 2
    assert custom.is_formal is False


@pytest.mark.parametrize("counts", [(40, 202, 404), (41, 200, 400)])
def test_mixed_formal_protocols_are_rejected_even_in_debug(counts) -> None:
    with pytest.raises(ValueError, match="mixed formal"):
        SamplingProtocol.from_counts(counts, allow_custom=True)


def test_pc_bacs_requires_target_counts_and_seed_mode() -> None:
    with pytest.raises(SystemExit):
        parse_pc_bacs_args(["--seed-mode", "legacy-kmeans"])
    with pytest.raises(SystemExit):
        parse_pc_bacs_args(["--target-counts", "41", "202", "404"])

    parsed = parse_pc_bacs_args(
        [
            "--target-counts",
            "41",
            "202",
            "404",
            "--seed-mode",
            "external-bootstrap",
        ]
    )
    assert parsed.target_counts == [41, 202, 404]
    assert parsed.seed_mode == "external-bootstrap"


def test_pc_bacs_seed_modes_have_distinct_count_and_membership_contracts() -> None:
    protocol = SamplingProtocol.from_counts((41, 202, 404))
    assert (
        _expected_seed_count(
            seed_mode="legacy-kmeans", n_clusters=40, protocol=protocol
        )
        == 40
    )
    assert (
        _expected_seed_count(
            seed_mode="external-bootstrap", n_clusters=40, protocol=protocol
        )
        == 41
    )

    with pytest.raises(ValueError, match="not the deterministic KMeans-center seed"):
        _validate_seed_for_mode(
            ["TR-CAMO/external"],
            ["TR-CAMO/center"],
            seed_mode="legacy-kmeans",
            label="selection seed",
        )
    _validate_seed_for_mode(
        ["TR-CAMO/external"],
        ["TR-CAMO/center"],
        seed_mode="external-bootstrap",
        label="selection seed",
    )


def test_split_artifacts_are_sorted_atomic_and_catalog_validated(tmp_path) -> None:
    split_path = tmp_path / "split.pt"
    fingerprint = save_split_keys(
        split_path, ["TR-COD10K/b", "TR-CAMO/a"]
    )
    assert load_split_keys(
        split_path,
        catalog_keys=["TR-CAMO/a", "TR-COD10K/b", "TR-COD10K/c"],
        expected_count=2,
    ) == ["TR-CAMO/a", "TR-COD10K/b"]
    assert isinstance(fingerprint, str) and len(fingerprint) == 64
    assert torch.load(split_path, weights_only=False) == [
        "TR-CAMO/a",
        "TR-COD10K/b",
    ]

    save_split_keys(split_path, ["TR-CAMO/a", "TR-COD10K/b"])
    with pytest.raises(FileExistsError):
        save_split_keys(split_path, ["TR-CAMO/a", "TR-COD10K/c"])
    with pytest.raises(ValueError, match="absent from the catalog"):
        load_split_keys(split_path, catalog_keys=["TR-CAMO/a"])


def test_shared_fingerprints_cover_order_catalog_and_image_content(tmp_path) -> None:
    keys = ["TR-CAMO/a", "TR-COD10K/b"]
    assert compute_catalog_fingerprint(keys) == compute_catalog_fingerprint(keys[::-1])
    assert compute_key_order_fingerprint(keys) != compute_key_order_fingerprint(keys[::-1])
    assert stable_fingerprint({"b": 2, "a": 1}) == stable_fingerprint(
        {"a": 1, "b": 2}
    )

    first = tmp_path / "a.jpg"
    second = tmp_path / "b.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    before = compute_image_fingerprint(keys, [first, second])
    second.write_bytes(b"changed")
    after = compute_image_fingerprint(keys, [first, second])
    assert before != after


def test_atomic_json_is_idempotent_and_refuses_mismatch(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    atomic_json_save({"protocol": [41, 202, 404]}, path)
    atomic_json_save({"protocol": [41, 202, 404]}, path)
    assert json.loads(path.read_text(encoding="utf-8"))["protocol"] == [41, 202, 404]
    with pytest.raises(FileExistsError):
        atomic_json_save({"protocol": [40, 200, 400]}, path)
