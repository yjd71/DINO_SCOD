from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from PIL import Image

import select_kmeans_only as cli
from utils.checkpoint_pc_hbm import compute_labeled_split_fingerprint
from utils.dataloader import SelectionPoolDataset


def _build_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, Path, list[str], list[str]]:
    data_root = tmp_path / "Dataset" / "COD"
    for subset, names in (("SET-A", ("a0", "a1")), ("SET-B", ("b0", "b1"))):
        image_root = data_root / subset / "im"
        image_root.mkdir(parents=True)
        for index, name in enumerate(names):
            Image.new(
                "RGB",
                (8, 8),
                color=(20 + index, 40 + index, 60 + index),
            ).save(image_root / f"{name}.jpg")

    weight_path = tmp_path / "dinov2_vitb14_pretrain.pth"
    weight_path.write_bytes(b"synthetic-dino-weight")
    monkeypatch.setattr(cli, "DINO_WEIGHT_PATH", weight_path)

    dataset = SelectionPoolDataset(
        [str(data_root / "SET-A" / "im"), str(data_root / "SET-B" / "im")]
    )
    sample_keys = list(dataset.sample_keys)
    catalog_fingerprint = cli._catalog_fingerprint(
        [dict(item) for item in dataset.items]
    )
    dino_fingerprint = cli._sha256_file(weight_path)
    static_spec = cli._expected_feature_static_spec(
        catalog_fingerprint=catalog_fingerprint,
        dino_fingerprint=dino_fingerprint,
    )
    full_spec = {
        **static_spec,
        "feature_amp": False,
        "feature_device_type": "cpu",
    }
    features = torch.zeros(4, 768, dtype=torch.float32)
    features[0, 0] = 1.0
    features[1, 0] = 1.0
    features[1, 1] = 0.1
    features[2, 1] = 1.0
    features[3, 1] = 1.0
    features[3, 0] = 0.1
    feature_payload = {
        **full_spec,
        "feature_spec_fingerprint": cli._sha256_json(full_spec),
        "normalized": False,
        "sample_keys": sample_keys,
        "features": features,
    }
    feature_path = data_root / "cache" / "features.pt"
    feature_path.parent.mkdir(parents=True)
    torch.save(feature_payload, feature_path)

    output_dir = data_root / "splits" / "kmeans_only"
    argv = [
        "--data-root",
        str(data_root),
        "--train-sets",
        "SET-A",
        "SET-B",
        "--features-path",
        str(feature_path),
        "--n-clusters",
        "2",
        "--seed",
        "2025",
        "--target-counts",
        "3",
        "4",
        "--output-dir",
        str(output_dir),
    ]
    return data_root, feature_path, output_dir, sample_keys, argv


def test_cli_source_has_no_selector_score_or_model_forward() -> None:
    cli_source = Path(cli.__file__).read_text(encoding="utf-8")
    algorithm_source = (
        Path(cli.__file__).parent / "utils" / "kmeans_only.py"
    ).read_text(encoding="utf-8")
    combined = cli_source + algorithm_source
    for forbidden in (
        "selector_checkpoint",
        "source_score_cache",
        "compute_pc_bacs_score",
        "score_pool",
        "BaseModel",
        "D_bd",
        "D_all",
        "dedup_threshold",
    ):
        assert forbidden not in combined


def test_cli_defaults_are_the_formal_protocol() -> None:
    args = cli.parse_args([])
    assert args.n_clusters == 40
    assert args.seed == 2025
    assert args.target_counts == [41, 202, 404, 808]
    assert args.features_path.as_posix().endswith(
        "Dataset/COD/cache/pc_bacs_dino_vitb14_392.pt"
    )
    assert args.output_dir.as_posix().endswith("Dataset/COD/splits/kmeans_only")


def test_feature_cache_validation_is_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, feature_path, _, sample_keys, _ = _build_inputs(tmp_path, monkeypatch)
    payload = torch.load(feature_path, map_location="cpu", weights_only=False)
    expected = cli._expected_feature_static_spec(
        catalog_fingerprint=str(payload["catalog_fingerprint"]),
        dino_fingerprint=str(payload["dino_fingerprint"]),
    )
    features, metadata = cli._validate_feature_cache(
        payload,
        sample_keys=sample_keys,
        expected_static_spec=expected,
    )
    assert features.dtype == torch.float32
    assert features.shape == (4, 768)
    assert metadata["feature_spec_fingerprint"] == payload[
        "feature_spec_fingerprint"
    ]

    invalid_payloads: list[tuple[dict[str, object], str]] = []
    wrong_order = dict(payload)
    wrong_order["sample_keys"] = list(reversed(sample_keys))
    invalid_payloads.append((wrong_order, "stable catalog order"))
    wrong_catalog = dict(payload)
    wrong_catalog["catalog_fingerprint"] = "wrong"
    invalid_payloads.append((wrong_catalog, "catalog_fingerprint"))
    wrong_spec = dict(payload)
    wrong_spec["feature_spec_fingerprint"] = "wrong"
    invalid_payloads.append((wrong_spec, "feature_spec_fingerprint"))
    wrong_dtype = dict(payload)
    wrong_dtype["features"] = payload["features"].half()
    invalid_payloads.append((wrong_dtype, "dtype"))
    wrong_shape = dict(payload)
    wrong_shape["features"] = payload["features"][:, :-1]
    invalid_payloads.append((wrong_shape, "shape"))
    nonfinite = dict(payload)
    nonfinite_features = payload["features"].clone()
    nonfinite_features[0, 0] = float("nan")
    nonfinite["features"] = nonfinite_features
    invalid_payloads.append((nonfinite, "NaN or Inf"))
    zero_norm = dict(payload)
    zero_features = payload["features"].clone()
    zero_features[0].zero_()
    zero_norm["features"] = zero_features
    invalid_payloads.append((zero_norm, "zero-norm"))

    for invalid, match in invalid_payloads:
        with pytest.raises(ValueError, match=match):
            cli._validate_feature_cache(
                invalid,
                sample_keys=sample_keys,
                expected_static_spec=expected,
            )


def test_dry_run_formal_artifacts_idempotence_and_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, _, output_dir, sample_keys, argv = _build_inputs(tmp_path, monkeypatch)

    assert cli.main([*argv, "--dry-run"]) == 0
    dry_report = json.loads(capsys.readouterr().out)
    assert dry_report["catalog_count"] == 4
    assert dry_report["n_clusters"] == 2
    assert dry_report["quota_policy"] == "sqrt_remaining_capacity_largest_remainder"
    assert dry_report["uses_selector"] is False
    assert dry_report["uses_score_cache"] is False
    assert dry_report["uses_pc_bacs_score"] is False
    assert dry_report["dedup"] is False
    assert not output_dir.exists()

    assert cli.main(argv) == 0
    capsys.readouterr()
    assert cli.main(argv) == 0
    capsys.readouterr()

    expected_labels = (2, 3, 4)
    previous: set[str] = set()
    for label in expected_labels:
        artifact_stem = f"kmeans_only_{label:04d}"
        if label == 2:
            artifact_stem += "_seed"
        pt_path = output_dir / f"{artifact_stem}_keys.pt"
        txt_path = output_dir / f"{artifact_stem}_labeled_names.txt"
        keys = torch.load(pt_path, map_location="cpu", weights_only=False)
        names = txt_path.read_text(encoding="utf-8").splitlines()
        assert isinstance(keys, list)
        assert keys == sorted(keys)
        assert len(keys) == len(set(keys)) == label
        assert names == [key.rsplit("/", 1)[-1] for key in keys]
        assert previous <= set(keys)
        previous = set(keys)

    with (output_dir / "kmeans_only_assignments.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == len(sample_keys)
    assert "score" not in {name.lower() for name in rows[0]}
    assert sum(int(row["is_center_seed"]) for row in rows) == 2
    assert sum(int(row["selected_0003"]) for row in rows) == 3
    assert sum(int(row["selected_0004"]) for row in rows) == 4

    manifest = json.loads(
        (output_dir / "kmeans_only_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["protocol"] == "kmeans_only_sqrt_quota_center_distance_v1"
    assert manifest["selection"]["uses_selector"] is False
    assert manifest["selection"]["uses_score_cache"] is False
    assert manifest["selection"]["uses_pc_bacs_score"] is False
    assert manifest["selection"]["dedup"] is False
    for label in expected_labels:
        artifact_stem = f"kmeans_only_{label:04d}"
        if label == 2:
            artifact_stem += "_seed"
        keys = torch.load(
            output_dir / f"{artifact_stem}_keys.pt",
            map_location="cpu",
            weights_only=False,
        )
        assert manifest["outputs"][str(label)][
            "split_fingerprint"
        ] == compute_labeled_split_fingerprint(keys)

    conflict_path = output_dir / "kmeans_only_0003_labeled_names.txt"
    conflict_path.write_text("different\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        cli.main(argv)


def test_cli_rejects_missing_cache_and_invalid_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, feature_path, _, _, argv = _build_inputs(tmp_path, monkeypatch)
    feature_path.unlink()
    with pytest.raises(FileNotFoundError, match="feature cache"):
        cli.main(argv)

    _, _, _, _, argv = _build_inputs(tmp_path / "second", monkeypatch)
    target_index = argv.index("--target-counts")
    invalid_argv = argv[: target_index + 1] + ["1", "4"] + argv[target_index + 3 :]
    with pytest.raises(ValueError, match="greater than --n-clusters"):
        cli.main(invalid_argv)
