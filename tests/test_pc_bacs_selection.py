from __future__ import annotations

import copy
import json

import pytest
import torch

from configs.pc_bacs_config import PCBACSConfig, SCORE_FORMULA_VERSION
from utils.pc_bacs import (
    allocate_cluster_quotas,
    atomic_json_save,
    build_feature_cache,
    build_nested_splits,
    build_score_cache,
    build_selection_manifest,
    compute_catalog_fingerprint,
    compute_image_fingerprint,
    compute_key_order_fingerprint,
    fit_dino_kmeans,
    load_split_keys,
    preprocessing_fingerprint,
    save_split_keys,
    stable_fingerprint,
    validate_feature_cache,
    validate_score_cache,
)


def test_config_defaults_match_confirmed_41_202_404_protocol() -> None:
    config = PCBACSConfig()

    assert config.target_counts == (41, 202, 404)
    assert config.selector_seed_count == 40
    assert config.n_clusters == 40
    assert config.random_seed == 2025
    assert config.dedup_threshold == pytest.approx(0.98)
    assert config.score_formula_version == SCORE_FORMULA_VERSION
    config.validate(sample_count=4040)


def test_config_rejects_invalid_protocol_values() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        PCBACSConfig(target_counts=(41, 41, 404)).validate()
    with pytest.raises(ValueError, match="selector_seed_count"):
        PCBACSConfig(selector_seed_count=42).validate()
    with pytest.raises(ValueError, match="largest target"):
        PCBACSConfig().validate(sample_count=403)
    with pytest.raises(ValueError, match="dedup_threshold"):
        PCBACSConfig(dedup_threshold=1.01).validate()


def test_largest_remainder_quota_is_exact_stable_and_capacity_bounded() -> None:
    assert allocate_cluster_quotas({0: 4, 1: 9}, 5) == {0: 2, 1: 3}
    assert allocate_cluster_quotas({0: 4, 1: 4}, 1) == {0: 1, 1: 0}
    assert allocate_cluster_quotas({0: 1, 1: 100}, 101) == {0: 1, 1: 100}
    assert allocate_cluster_quotas({2: 0, 0: 3}, 0) == {0: 0, 2: 0}

    with pytest.raises(ValueError, match="exceeds"):
        allocate_cluster_quotas({0: 1}, 2)


def test_kmeans_is_fixed_and_center_ties_use_sample_key() -> None:
    keys = ["TR-CAMO/z", "TR-CAMO/a", "TR-COD10K/y", "TR-COD10K/b"]
    features = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    )

    first = fit_dino_kmeans(keys, features, n_clusters=2, random_seed=2025)
    second = fit_dino_kmeans(keys, features, n_clusters=2, random_seed=2025)

    assert set(first.seed_keys) == {"TR-CAMO/a", "TR-COD10K/b"}
    assert first.seed_keys == second.seed_keys
    torch.testing.assert_close(first.cluster_ids, second.cluster_ids)
    torch.testing.assert_close(first.center_distances, torch.zeros(4))
    torch.testing.assert_close(
        first.normalized_features.norm(dim=1), torch.ones(4), rtol=0.0, atol=1e-7
    )


def test_kmeans_rejects_too_few_distinct_features() -> None:
    with pytest.raises(ValueError, match="distinct features"):
        fit_dino_kmeans(
            ["TR-CAMO/a", "TR-CAMO/b"],
            torch.ones(2, 4),
            n_clusters=2,
        )


def test_default_nested_counts_are_exact_and_reproducible() -> None:
    generator = torch.Generator().manual_seed(2025)
    sample_count = 500
    keys = [f"TR-CAMO/sample_{index:04d}" for index in range(sample_count)]
    features = torch.randn(sample_count, 12, generator=generator)
    cluster_ids = torch.arange(sample_count) % 40
    scores = torch.linspace(1.0, 0.0, sample_count)
    seed_keys = keys[:40]

    first = build_nested_splits(
        keys,
        features,
        cluster_ids,
        scores,
        seed_keys,
        dedup_threshold=-1.0,
    )
    second = build_nested_splits(
        keys,
        features,
        cluster_ids,
        scores,
        seed_keys,
        dedup_threshold=-1.0,
    )

    assert {target: len(split) for target, split in first.splits.items()} == {
        41: 41,
        202: 202,
        404: 404,
    }
    assert set(first.splits[41]) < set(first.splits[202]) < set(first.splits[404])
    assert first.splits == second.splits
    assert first.selection_order == second.selection_order
    assert set(first.selection_rank) == {41, 202, 404}
    assert set(first.selection_rank[41]) == set(first.splits[41])


def test_score_ties_use_lexical_sample_key() -> None:
    keys = ["TR-CAMO/z", "TR-CAMO/c", "TR-CAMO/a", "TR-CAMO/b"]
    features = torch.eye(4)

    result = build_nested_splits(
        keys,
        features,
        torch.zeros(4, dtype=torch.long),
        torch.ones(4),
        ["TR-CAMO/z"],
        target_counts=(2,),
        dedup_threshold=-1.0,
    )

    assert result.splits[2] == ["TR-CAMO/a", "TR-CAMO/z"]


def test_dedup_uses_seed_then_global_and_relaxed_backfill() -> None:
    keys = [
        "TR-CAMO/seed0",
        "TR-CAMO/c0_a",
        "TR-CAMO/c0_b",
        "TR-CAMO/c0_c",
        "TR-CAMO/c0_d",
        "TR-CAMO/seed1",
        "TR-CAMO/c1_a",
        "TR-CAMO/c1_b",
        "TR-CAMO/c1_c",
        "TR-CAMO/c1_d",
    ]
    features = torch.zeros(10, 8)
    features[:5, 0] = 1.0
    for offset, index in enumerate(range(5, 10), start=1):
        features[index, offset] = 1.0
    cluster_ids = torch.tensor([0] * 5 + [1] * 5)
    scores = torch.tensor([0.0, 0.99, 0.98, 0.97, 0.96, 0.0, 0.5, 0.4, 0.3, 0.2])

    result = build_nested_splits(
        keys,
        features,
        cluster_ids,
        scores,
        ["TR-CAMO/seed0", "TR-CAMO/seed1"],
        target_counts=(9,),
        dedup_threshold=0.98,
    )

    round_stats = result.rounds[0]
    assert len(result.splits[9]) == 9
    assert round_stats["quota_selected_count"] == 3
    assert round_stats["dedup_backfill_count"] == 1
    assert round_stats["relaxed_backfill_count"] == 3
    assert round_stats["dedup_skips"] > 0
    assert sum(result.dedup_skipped_count[key] for key in keys[1:5]) > 0


def test_split_atomic_save_is_idempotent_and_refuses_different_content(tmp_path) -> None:
    path = tmp_path / "pc_bacs_0041_keys.pt"
    fingerprint = save_split_keys(path, ["TR-COD10K/b", "TR-CAMO/a"])

    assert load_split_keys(path) == ["TR-CAMO/a", "TR-COD10K/b"]
    assert save_split_keys(path, ["TR-CAMO/a", "TR-COD10K/b"]) == fingerprint
    with pytest.raises(FileExistsError, match="Refusing"):
        save_split_keys(path, ["TR-CAMO/a", "TR-COD10K/c"])
    assert load_split_keys(path) == ["TR-CAMO/a", "TR-COD10K/b"]


def test_feature_cache_strictly_checks_identity_and_key_order() -> None:
    keys = ["TR-CAMO/a", "TR-COD10K/b"]
    features = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    cache = build_feature_cache(
        keys,
        features,
        catalog_fingerprint="catalog",
        image_fingerprint="images",
        dino_weight_fingerprint="weight",
        preprocessing_fingerprint="preprocess",
    )

    loaded_keys, loaded_features = validate_feature_cache(
        cache,
        expected_sample_keys=keys,
        expected_catalog_fingerprint="catalog",
        expected_image_fingerprint="images",
        expected_dino_weight_fingerprint="weight",
        expected_preprocessing_fingerprint="preprocess",
        feature_dim=3,
    )
    assert loaded_keys == keys
    torch.testing.assert_close(loaded_features, features)

    tampered = copy.deepcopy(cache)
    tampered["sample_keys"] = list(reversed(keys))
    tampered["key_order_fingerprint"] = compute_key_order_fingerprint(tampered["sample_keys"])
    with pytest.raises(ValueError, match="order"):
        validate_feature_cache(
            tampered,
            expected_sample_keys=keys,
            expected_catalog_fingerprint="catalog",
            expected_image_fingerprint="images",
            expected_dino_weight_fingerprint="weight",
            expected_preprocessing_fingerprint="preprocess",
            feature_dim=3,
        )


def test_score_cache_strictly_checks_formula_and_selector_identity() -> None:
    keys = ["TR-CAMO/a", "TR-COD10K/b"]
    boundary = torch.tensor([0.5, 0.25])
    global_value = torch.tensor([0.2, 0.4])
    scores = boundary * (1.0 - global_value)
    cache = build_score_cache(
        keys,
        boundary,
        global_value,
        scores,
        selector_fingerprint="selector",
        catalog_fingerprint="catalog",
        image_fingerprint="images",
        preprocessing_fingerprint="preprocess",
    )

    loaded = validate_score_cache(
        cache,
        expected_sample_keys=keys,
        expected_selector_fingerprint="selector",
        expected_catalog_fingerprint="catalog",
        expected_image_fingerprint="images",
        expected_preprocessing_fingerprint="preprocess",
    )
    torch.testing.assert_close(loaded["scores"], scores)

    with pytest.raises(ValueError, match="selector_fingerprint"):
        validate_score_cache(
            cache,
            expected_sample_keys=keys,
            expected_selector_fingerprint="different",
            expected_catalog_fingerprint="catalog",
            expected_image_fingerprint="images",
            expected_preprocessing_fingerprint="preprocess",
        )


def test_fingerprints_cover_catalog_order_and_image_content(tmp_path) -> None:
    keys = ["TR-CAMO/a", "TR-COD10K/b"]
    first = tmp_path / "a.jpg"
    second = tmp_path / "b.jpg"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    catalog = compute_catalog_fingerprint(keys)
    assert catalog == compute_catalog_fingerprint(list(reversed(keys)))
    assert compute_key_order_fingerprint(keys) != compute_key_order_fingerprint(list(reversed(keys)))
    image_fingerprint = compute_image_fingerprint(keys, [first, second])
    first.write_bytes(b"changed")
    assert image_fingerprint != compute_image_fingerprint(keys, [first, second])
    assert preprocessing_fingerprint() == preprocessing_fingerprint(392)
    assert stable_fingerprint({"b": 2, "a": 1}) == stable_fingerprint({"a": 1, "b": 2})


def test_manifest_is_json_serializable_and_atomic(tmp_path) -> None:
    manifest = build_selection_manifest(
        config=PCBACSConfig(),
        dataset={"sample_count": 4040, "catalog_fingerprint": "catalog"},
        selector={"non_pc_fingerprint": "selector", "epochs": 5},
        outputs={41: {"path": "pc_bacs_0041_keys.pt", "fingerprint": "split"}},
        repo_commit="df4d610",
        runtime={"python": "test"},
        fingerprints={"images": "images"},
    )

    assert manifest["selection"]["score_formula_version"] == SCORE_FORMULA_VERSION
    json.dumps(manifest, allow_nan=False)
    path = tmp_path / "pc_bacs_manifest.json"
    atomic_json_save(manifest, path)
    atomic_json_save(manifest, path)
    assert json.loads(path.read_text(encoding="utf-8")) == manifest

    different = copy.deepcopy(manifest)
    different["repo_commit"] = "different"
    with pytest.raises(FileExistsError, match="Refusing"):
        atomic_json_save(different, path)
