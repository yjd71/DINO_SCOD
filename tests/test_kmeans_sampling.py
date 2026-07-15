from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import kmeans_sample as rdsm
from selection.artifacts import atomic_torch_save, load_split_keys, save_split_keys
from selection.protocol import SamplingProtocol


def _toy_pool() -> tuple[list[str], np.ndarray]:
    keys = [f"TR-CAMO/sample-{index:02d}" for index in range(10)]
    features = np.asarray(
        [
            [-4.0, -0.2],
            [-4.1, 0.1],
            [-3.9, 0.3],
            [-0.2, 4.0],
            [0.1, 4.1],
            [0.3, 3.9],
            [4.0, -0.2],
            [4.1, 0.1],
            [3.9, 0.3],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )
    return keys, features


def test_cli_requires_mode_and_explicit_target_counts() -> None:
    with pytest.raises(SystemExit):
        rdsm.parse_args([])
    with pytest.raises(SystemExit):
        rdsm.parse_args(["--mode", "original"])
    with pytest.raises(SystemExit):
        rdsm.parse_args(["--target-counts", "40", "200", "400"])

    args = rdsm.parse_args(
        ["--mode", "original", "--target-counts", "41", "202", "404"]
    )
    assert args.target_counts == [41, 202, 404]
    assert rdsm.resolve_protocol(args).name == "scout_0041_0202_0404"


@pytest.mark.parametrize("counts", [(40, 200, 400), (41, 202, 404)])
def test_both_formal_protocols_are_accepted(counts: tuple[int, int, int]) -> None:
    args = rdsm.parse_args(
        ["--mode", "original", "--target-counts", *map(str, counts)]
    )
    protocol = rdsm.resolve_protocol(args)
    assert protocol.target_counts == counts
    assert protocol.is_formal


def test_custom_counts_require_debug_switch_and_isolated_output(tmp_path: Path) -> None:
    base = ["--mode", "original", "--target-counts", "2", "4", "6"]
    with pytest.raises(ValueError, match="unsupported target-count protocol"):
        rdsm.resolve_protocol(rdsm.parse_args(base))

    args = rdsm.parse_args(
        [*base, "--debug-custom-counts", "--data-root", str(tmp_path)]
    )
    protocol = rdsm.resolve_protocol(args)
    output_dir, _ = rdsm._resolve_paths(args, protocol)
    assert not protocol.is_formal
    assert output_dir.is_relative_to(tmp_path / "splits" / "rdsm" / "debug")

    args.output_dir = tmp_path / "formal-looking-output"
    with pytest.raises(ValueError, match="isolated debug root"):
        rdsm._resolve_paths(args, protocol)


@pytest.mark.parametrize(
    "counts",
    [
        (41, 41, 404),
        (202, 41, 404),
        (40, 202, 404),
        (0, 2, 4),
    ],
)
def test_malformed_or_mixed_protocols_fail(counts: tuple[int, int, int]) -> None:
    args = rdsm.parse_args(
        ["--mode", "original", "--target-counts", *map(str, counts)]
    )
    with pytest.raises((TypeError, ValueError)):
        rdsm.resolve_protocol(args)


def test_mixed_formal_protocol_fails_even_with_debug_switch() -> None:
    args = rdsm.parse_args(
        [
            "--mode",
            "original",
            "--target-counts",
            "40",
            "202",
            "404",
            "--debug-custom-counts",
        ]
    )
    with pytest.raises(ValueError, match="mixed formal"):
        rdsm.resolve_protocol(args)


def test_kmeans_configuration_is_fully_locked(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeKMeans:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        def fit(self, features: np.ndarray) -> "FakeKMeans":
            captured["features"] = features.copy()
            return self

    monkeypatch.setattr(rdsm, "KMeans", FakeKMeans)
    features = np.arange(12, dtype=np.float32).reshape(6, 2)
    fitted = rdsm._fit_kmeans(features, 3, 2027)

    assert isinstance(fitted, FakeKMeans)
    assert captured["n_clusters"] == 3
    assert captured["random_state"] == 2027
    assert captured["n_init"] == 10
    assert captured["algorithm"] == "lloyd"
    np.testing.assert_array_equal(captured["features"], features)


def test_original_is_exact_deterministic_and_order_invariant() -> None:
    keys, features = _toy_pool()
    counts = (2, 5, 8)
    first = rdsm.select_rdsm_original(keys, features, counts, seed=2025)
    second = rdsm.select_rdsm_original(keys, features, counts, seed=2025)

    permutation = [7, 1, 9, 3, 0, 6, 2, 8, 4, 5]
    shuffled = rdsm.select_rdsm_original(
        [keys[index] for index in permutation],
        features[permutation],
        counts,
        seed=2025,
    )

    assert first == second == shuffled
    assert first.fitted_cluster_counts == counts
    assert first.acquisition_order == ()
    for count in counts:
        split = first.splits[count]
        assert len(split) == count
        assert len(set(split)) == count
        assert list(split) == sorted(split)


def test_original_remains_exact_for_duplicate_feature_degeneracy() -> None:
    keys = [f"sample-{index}" for index in range(6)]
    features = np.zeros((6, 3), dtype=np.float32)

    with pytest.warns(Warning, match="distinct clusters"):
        result = rdsm.select_rdsm_original(keys, features, (2, 4, 6), seed=2025)

    assert {count: len(split) for count, split in result.splits.items()} == {
        2: 2,
        4: 4,
        6: 6,
    }


def test_seeded_uses_one_fit_and_builds_strict_nested_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys, features = _toy_pool()
    bootstrap = [keys[7], keys[1]]
    real_fit = rdsm._fit_kmeans
    fitted_counts: list[int] = []

    def recording_fit(array: np.ndarray, n_clusters: int, seed: int):
        fitted_counts.append(n_clusters)
        return real_fit(array, n_clusters, seed)

    monkeypatch.setattr(rdsm, "_fit_kmeans", recording_fit)
    result = rdsm.select_rdsm_seeded(
        keys, features, (2, 5, 8), bootstrap, seed=2026
    )

    assert fitted_counts == [2]
    assert list(result.splits[2]) == sorted(bootstrap)
    assert set(result.splits[2]) < set(result.splits[5]) < set(result.splits[8])
    assert [len(result.splits[count]) for count in (2, 5, 8)] == [2, 5, 8]
    assert len(result.acquisition_order) == 6
    assert not (set(result.acquisition_order) & set(bootstrap))


def test_seeded_is_catalog_order_invariant() -> None:
    keys, features = _toy_pool()
    baseline = rdsm.select_rdsm_seeded(
        keys, features, (2, 5, 8), [keys[1], keys[7]], seed=2027
    )
    permutation = [9, 5, 1, 4, 8, 2, 7, 3, 0, 6]
    shuffled = rdsm.select_rdsm_seeded(
        [keys[index] for index in permutation],
        features[permutation],
        (2, 5, 8),
        [keys[7], keys[1]],
        seed=2027,
    )
    assert shuffled == baseline


def test_seeded_cluster_queues_tie_break_by_key_and_ignore_label_permutation() -> None:
    keys = ["boot-a", "a", "b", "boot-z", "c", "d"]
    distances = np.asarray([0.0, 0.1, 0.1, 0.0, 0.1, 0.1])
    bootstrap = ["boot-a", "boot-z"]

    first = rdsm._seeded_acquisition_order(
        keys, np.asarray([0, 0, 0, 1, 1, 1]), distances, bootstrap
    )
    relabeled = rdsm._seeded_acquisition_order(
        keys, np.asarray([9, 9, 9, 3, 3, 3]), distances, bootstrap
    )

    assert first == ["a", "c", "b", "d"]
    assert relabeled == first


def test_seeded_validates_bootstrap_exactly() -> None:
    keys, features = _toy_pool()
    with pytest.raises(ValueError, match="exactly 2 unique"):
        rdsm.select_rdsm_seeded(
            keys, features, (2, 5, 8), [keys[0]], seed=2025
        )
    with pytest.raises(ValueError, match="absent from the catalog"):
        rdsm.select_rdsm_seeded(
            keys, features, (2, 5, 8), [keys[0], "missing"], seed=2025
        )


def test_feature_cache_validation_binds_stable_key_order_and_spec() -> None:
    keys = ["TR-CAMO/a", "TR-COD10K/b"]
    spec = {"schema": rdsm.FEATURE_CACHE_SCHEMA, "identity": "expected"}
    payload = {
        "spec": dict(spec),
        "sample_keys": list(keys),
        "features": torch.ones(2, 3, dtype=torch.float32),
    }
    loaded = rdsm._validate_feature_cache(
        payload, expected_spec=spec, sample_keys=keys
    )
    torch.testing.assert_close(loaded, payload["features"])

    with pytest.raises(ValueError, match="sample-key order mismatch"):
        rdsm._validate_feature_cache(
            payload, expected_spec=spec, sample_keys=list(reversed(keys))
        )
    with pytest.raises(ValueError, match="identity mismatch"):
        rdsm._validate_feature_cache(
            payload,
            expected_spec={"schema": rdsm.FEATURE_CACHE_SCHEMA, "identity": "other"},
            sample_keys=keys,
        )


def test_seeded_artifacts_are_plain_sorted_lists_with_auditable_manifest(
    tmp_path: Path,
) -> None:
    keys, features = _toy_pool()
    counts = (2, 5, 8)
    bootstrap = sorted([keys[1], keys[7]])
    result = rdsm.select_rdsm_seeded(
        keys, features, counts, bootstrap, seed=2025
    )
    protocol = SamplingProtocol.from_counts(counts, allow_custom=True)
    cache_path = tmp_path / "features.pt"
    bootstrap_path = tmp_path / "bootstrap.pt"
    output_dir = tmp_path / "outputs"
    atomic_torch_save({"features": torch.from_numpy(features)}, cache_path)
    save_split_keys(bootstrap_path, bootstrap)

    manifest = rdsm._save_selection_artifacts(
        result=result,
        output_dir=output_dir,
        protocol=protocol,
        seed=2025,
        sample_keys=keys,
        catalog_fingerprint="catalog-fingerprint",
        image_fingerprint="image-fingerprint",
        cache_path=cache_path,
        feature_spec={"schema": "test"},
        bootstrap_path=bootstrap_path,
        bootstrap_keys=bootstrap,
    )

    assert manifest["selection"]["strictly_nested"] is True
    assert manifest["kmeans"]["fitted_cluster_counts"] == [2]
    assert manifest["kmeans"]["n_init"] == 10
    assert manifest["kmeans"]["algorithm"] == "lloyd"
    for count in counts:
        split_path = output_dir / f"rdsm_seeded_{count:04d}_seed2025.pt"
        split = load_split_keys(split_path, catalog_keys=keys, expected_count=count)
        assert split == sorted(split)
        assert isinstance(torch.load(split_path, weights_only=False), list)

    stored_manifest = json.loads(
        (output_dir / "selection_manifest.json").read_text(encoding="utf-8")
    )
    assert stored_manifest == manifest


def test_original_rejects_bootstrap_cli_argument() -> None:
    args = rdsm.parse_args(
        [
            "--mode",
            "original",
            "--target-counts",
            "40",
            "200",
            "400",
            "--bootstrap-split",
            "bootstrap.pt",
        ]
    )
    with pytest.raises(ValueError, match="only valid for seeded"):
        rdsm._validate_cli_args(args, rdsm.resolve_protocol(args))


def test_seeded_requires_bootstrap_cli_argument() -> None:
    args = rdsm.parse_args(
        ["--mode", "seeded", "--target-counts", "41", "202", "404"]
    )
    with pytest.raises(ValueError, match="required for seeded"):
        rdsm._validate_cli_args(args, rdsm.resolve_protocol(args))
