from __future__ import annotations

import json

import pytest

from selection.artifacts import load_split_keys
from selection.protocol import SamplingProtocol
from tools.make_bootstrap_split import (
    build_stratified_bootstrap,
    formal_subset_quotas,
    main,
)


def _catalog(camo: int, cod10k: int) -> list[str]:
    return sorted(
        [f"TR-CAMO/camo_{index:04d}" for index in range(camo)]
        + [f"TR-COD10K/cod_{index:04d}" for index in range(cod10k)]
    )


def _make_data_root(tmp_path, *, camo: int, cod10k: int):
    data_root = tmp_path / "COD"
    for subset, count in (("TR-CAMO", camo), ("TR-COD10K", cod10k)):
        image_root = data_root / subset / "im"
        image_root.mkdir(parents=True)
        for index in range(count):
            (image_root / f"image_{index:04d}.jpg").write_bytes(b"")
    return data_root


def test_formal_bootstrap_quotas_match_both_comparison_protocols() -> None:
    assert formal_subset_quotas(SamplingProtocol.from_counts((40, 200, 400))) == {
        "TR-CAMO": 10,
        "TR-COD10K": 30,
    }
    assert formal_subset_quotas(SamplingProtocol.from_counts((41, 202, 404))) == {
        "TR-CAMO": 10,
        "TR-COD10K": 31,
    }


def test_stratified_bootstrap_is_stable_sorted_and_seeded() -> None:
    catalog = _catalog(20, 60)
    quotas = {"TR-CAMO": 4, "TR-COD10K": 8}
    first = build_stratified_bootstrap(catalog, seed=2025, subset_quotas=quotas)
    repeated = build_stratified_bootstrap(catalog, seed=2025, subset_quotas=quotas)
    other_seed = build_stratified_bootstrap(catalog, seed=2026, subset_quotas=quotas)
    assert first == repeated
    assert first == sorted(first)
    assert first != other_seed
    assert sum(key.startswith("TR-CAMO/") for key in first) == 4
    assert sum(key.startswith("TR-COD10K/") for key in first) == 8


def test_bootstrap_rejects_unsorted_catalog_and_insufficient_subset() -> None:
    with pytest.raises(ValueError, match="sorted"):
        build_stratified_bootstrap(
            ["TR-COD10K/b", "TR-CAMO/a"],
            seed=1,
            subset_quotas={"TR-CAMO": 1},
        )
    with pytest.raises(ValueError, match="needs 2"):
        build_stratified_bootstrap(
            ["TR-CAMO/a"], seed=1, subset_quotas={"TR-CAMO": 2}
        )


def test_custom_bootstrap_cli_requires_debug_quota_and_isolated_output(tmp_path) -> None:
    data_root = _make_data_root(tmp_path, camo=3, cod10k=3)
    common = [
        "--data-root",
        str(data_root),
        "--target-counts",
        "2",
        "4",
        "6",
        "--seed",
        "2025",
        "--debug-allow-custom-counts",
        "--subset-quotas",
        "1",
        "1",
    ]
    with pytest.raises(ValueError, match="explicit isolated"):
        main(common)
    with pytest.raises(ValueError, match="outside"):
        main(common + ["--output-dir", str(data_root / "splits" / "debug")])

    output_dir = tmp_path / "debug-bootstrap"
    assert main(common + ["--output-dir", str(output_dir)]) == 0
    assert load_split_keys(
        output_dir / "bootstrap_0002_seed2025.pt", expected_count=2
    )
    manifest = json.loads(
        (output_dir / "bootstrap_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["protocol"]["is_formal"] is False
    assert manifest["subset_quotas"] == {"TR-CAMO": 1, "TR-COD10K": 1}


def test_formal_scout_bootstrap_cli_writes_valid_manifest_and_split(tmp_path) -> None:
    data_root = _make_data_root(tmp_path, camo=10, cod10k=394)
    output_dir = tmp_path / "formal-bootstrap"
    assert (
        main(
            [
                "--data-root",
                str(data_root),
                "--target-counts",
                "41",
                "202",
                "404",
                "--seed",
                "2025",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 0
    )
    split = load_split_keys(
        output_dir / "bootstrap_0041_seed2025.pt", expected_count=41
    )
    assert sum(key.startswith("TR-CAMO/") for key in split) == 10
    assert sum(key.startswith("TR-COD10K/") for key in split) == 31
    manifest = json.loads(
        (output_dir / "bootstrap_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["catalog"]["count"] == 404
    assert manifest["protocol"]["name"] == "scout_0041_0202_0404"
    assert manifest["split"]["count"] == 41
