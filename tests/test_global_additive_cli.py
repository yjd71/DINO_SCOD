from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from torch import nn

import select_global_additive as cli
from configs.pc_hbm_dino_config import DinoPCHBMConfig
from utils.checkpoint_pc_hbm import (
    build_artifact_metadata,
    compute_labeled_split_fingerprint,
    save_decoder_checkpoint,
)
from utils.dataloader import SelectionPoolDataset


class _TinyLegacyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(2, 1, kernel_size=1, bias=True)


def _build_synthetic_inputs(tmp_path: Path, monkeypatch):
    data_root = tmp_path / "Dataset" / "COD"
    image_root = data_root / "TR-CAMO" / "im"
    image_root.mkdir(parents=True)
    for index in range(4):
        (image_root / f"image_{index:02d}.jpg").write_bytes(
            f"synthetic-image-{index}".encode("ascii")
        )

    dataset = SelectionPoolDataset([str(image_root)])
    items = [dict(item) for item in dataset.items]
    sample_keys = list(dataset.sample_keys)
    seed_keys = [sample_keys[0]]
    seed_path = tmp_path / "seed.pt"
    torch.save(seed_keys, seed_path)

    checkpoint_path = tmp_path / "teacher_enhancer.pth"
    metadata = build_artifact_metadata(
        training_design="two_stage",
        artifact_role="teacher_enhancer",
        labeled_split_fingerprint=compute_labeled_split_fingerprint(seed_keys),
        baseline_fingerprint="synthetic-baseline",
        pc_frozen=True,
    )
    save_decoder_checkpoint(
        checkpoint_path,
        _TinyLegacyDecoder(),
        pc_cfg=DinoPCHBMConfig(),
        epoch=5,
        artifact_meta=metadata,
    )

    dino_path = tmp_path / "dino.pth"
    dino_path.write_bytes(b"synthetic-dino-weight")
    monkeypatch.setattr(cli, "DINO_WEIGHT_PATH", dino_path)
    dino_fingerprint = cli._sha256_file(dino_path)
    _, _, selector_fingerprint = cli._load_selector_identity(
        checkpoint_path,
        selector_seed_keys=seed_keys,
        dino_fingerprint=dino_fingerprint,
    )
    catalog_fingerprint = cli._catalog_fingerprint(items)
    spec = cli._expected_source_spec(
        catalog_fingerprint=catalog_fingerprint,
        selector_fingerprint=selector_fingerprint,
        dino_fingerprint=dino_fingerprint,
    )
    boundary = torch.tensor([0.0, 0.8, 0.4, 0.2], dtype=torch.float32)
    global_disagreement = torch.tensor([0.0, 0.1, 0.05, 0.0])
    source_payload = {
        **spec,
        "sample_keys": sample_keys,
        "boundary_disagreement": boundary,
        "global_disagreement": global_disagreement,
        "scores": boundary * (1.0 - global_disagreement),
    }
    source_path = tmp_path / "source_scores.pt"
    torch.save(source_payload, source_path)
    output_dir = tmp_path / "outputs"
    argv = [
        "--data-root",
        str(data_root),
        "--train-sets",
        "TR-CAMO",
        "--seed-split",
        str(seed_path),
        "--selector-checkpoint",
        str(checkpoint_path),
        "--source-score-cache",
        str(source_path),
        "--target-counts",
        "2",
        "3",
        "4",
        "--output-dir",
        str(output_dir),
    ]
    return data_root, output_dir, sample_keys, argv, source_payload


def _add_dedup_inputs(
    tmp_path: Path,
    argv: list[str],
    sample_keys: list[str],
    *,
    features: torch.Tensor | None = None,
) -> tuple[list[str], Path, dict[str, object]]:
    if features is None:
        features = torch.zeros(len(sample_keys), 768, dtype=torch.float32)
        # Highest-ranked candidate duplicates the seed. Candidate 2 is novel;
        # candidate 3 duplicates candidate 2, forcing deterministic relaxed fill.
        features[0, 0] = 1.0
        features[1, 0] = 1.0
        features[2, 1] = 1.0
        features[3, 1] = 1.0

    data_root = Path(argv[argv.index("--data-root") + 1])
    image_root = data_root / "TR-CAMO" / "im"
    dataset = SelectionPoolDataset([str(image_root)])
    catalog_fingerprint = cli._catalog_fingerprint(
        [dict(item) for item in dataset.items]
    )
    dino_fingerprint = cli._sha256_file(cli.DINO_WEIGHT_PATH)
    static_spec = cli._expected_dino_feature_static_spec(
        catalog_fingerprint=catalog_fingerprint,
        dino_fingerprint=dino_fingerprint,
    )
    full_spec = {
        **static_spec,
        "feature_amp": True,
        "feature_device_type": "cuda",
    }
    payload: dict[str, object] = {
        **full_spec,
        "feature_spec_fingerprint": cli._sha256_json(full_spec),
        "sample_keys": list(sample_keys),
        "features": features,
        "normalized": False,
        "weight_path": "weight/dinov2_vitb14_pretrain.pth",
    }
    feature_path = tmp_path / "dino_features.pt"
    torch.save(payload, feature_path)
    dedup_argv = [
        *argv,
        "--dedup-mode",
        "dino-cosine",
        "--dino-feature-cache",
        str(feature_path),
        "--dedup-threshold",
        "0.95",
    ]
    return dedup_argv, feature_path, payload


def test_cli_source_does_not_import_or_call_kmeans() -> None:
    source = (Path(__file__).parents[1] / "select_global_additive.py").read_text(
        encoding="utf-8"
    )
    assert "sklearn" not in source
    assert "KMeans(" not in source
    assert "fit_dino_kmeans" not in source
    assert "BaseModel" not in source


def test_dedup_cli_defaults_preserve_original_mode() -> None:
    args = cli.parse_args(
        [
            "--seed-split",
            "seed.pt",
            "--selector-checkpoint",
            "selector.pth",
            "--source-score-cache",
            "scores.pt",
            "--output-dir",
            "outputs",
        ]
    )

    assert args.dedup_mode == "none"
    assert args.dino_feature_cache is None
    assert args.dedup_threshold == pytest.approx(0.95)


def test_source_cache_rejects_wrong_formula_relation() -> None:
    expected = {"format_version": 1}
    payload = {
        "format_version": 1,
        "sample_keys": ["TR-CAMO/a"],
        "boundary_disagreement": torch.tensor([0.4]),
        "global_disagreement": torch.tensor([0.2]),
        "scores": torch.tensor([0.9]),
    }
    with pytest.raises(ValueError, match="do not satisfy"):
        cli._validate_source_score_cache(
            payload,
            sample_keys=["TR-CAMO/a"],
            expected_spec=expected,
        )


@pytest.mark.parametrize(
    ("field", "dtype"),
    [
        (field, dtype)
        for field in (
            "boundary_disagreement",
            "global_disagreement",
            "scores",
        )
        for dtype in (torch.float16, torch.float64)
    ],
)
def test_source_cache_rejects_non_float32_tensors(
    field: str, dtype: torch.dtype
) -> None:
    boundary = torch.tensor([0.4], dtype=torch.float32)
    global_disagreement = torch.tensor([0.2], dtype=torch.float32)
    payload = {
        "format_version": 1,
        "sample_keys": ["TR-CAMO/a"],
        "boundary_disagreement": boundary,
        "global_disagreement": global_disagreement,
        "scores": boundary * (1.0 - global_disagreement),
    }
    payload[field] = payload[field].to(dtype=dtype)

    with pytest.raises(ValueError, match=rf"{field} dtype must be torch.float32"):
        cli._validate_source_score_cache(
            payload,
            sample_keys=["TR-CAMO/a"],
            expected_spec={"format_version": 1},
        )


def test_selector_identity_accepts_strict_legacy_metadata_v1(
    tmp_path: Path, monkeypatch
) -> None:
    _, _, seed_keys, _, _ = _build_synthetic_inputs(tmp_path, monkeypatch)
    checkpoint_path = tmp_path / "teacher_enhancer.pth"
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = payload["artifact_meta"]
    metadata.pop("architecture", None)
    metadata.pop("schema_version", None)
    metadata["artifact_metadata_version"] = 1
    torch.save(payload, checkpoint_path)

    _, _, selector_fingerprint = cli._load_selector_identity(
        checkpoint_path,
        selector_seed_keys=[seed_keys[0]],
        dino_fingerprint=cli._sha256_file(tmp_path / "dino.pth"),
    )
    assert len(selector_fingerprint) == 64


def test_cli_dry_run_writes_nothing_and_formal_run_is_idempotent(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    data_root, output_dir, sample_keys, argv, _ = _build_synthetic_inputs(
        tmp_path, monkeypatch
    )

    assert cli.main([*argv, "--dry-run"]) == 0
    assert not output_dir.exists()
    assert not (data_root / "cache" / "global_additive_scores_split0.01.pt").exists()
    dry_output = capsys.readouterr().out
    assert '"uses_kmeans": false' in dry_output
    assert sample_keys[1] in dry_output

    assert cli.main(argv) == 0
    assert cli.main(argv) == 0
    expected_counts = {2: 2, 3: 3, 4: 4}
    for target, count in expected_counts.items():
        pt_path = output_dir / f"global_additive_{target:04d}_keys.pt"
        txt_path = output_dir / f"global_additive_{target:04d}_labeled_names.txt"
        keys = torch.load(pt_path, map_location="cpu", weights_only=False)
        names = txt_path.read_text(encoding="utf-8").splitlines()
        assert len(keys) == count
        assert names == [key.rsplit("/", 1)[-1] for key in keys]

    changed = output_dir / "global_additive_0002_labeled_names.txt"
    changed.write_text("different\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        cli.main(argv)


def test_source_cache_identity_mismatch_is_a_hard_failure(
    tmp_path: Path, monkeypatch
) -> None:
    _, output_dir, _, argv, payload = _build_synthetic_inputs(tmp_path, monkeypatch)
    payload["catalog_fingerprint"] = "wrong-catalog"
    source_path = Path(argv[argv.index("--source-score-cache") + 1])
    torch.save(payload, source_path)

    with pytest.raises(ValueError, match="catalog_fingerprint"):
        cli.main(argv)
    assert not output_dir.exists()


def test_dino_feature_cache_is_strictly_validated(
    tmp_path: Path, monkeypatch
) -> None:
    _, _, sample_keys, argv, _ = _build_synthetic_inputs(tmp_path, monkeypatch)
    _, _, payload = _add_dedup_inputs(tmp_path, argv, sample_keys)
    expected = cli._expected_dino_feature_static_spec(
        catalog_fingerprint=str(payload["catalog_fingerprint"]),
        dino_fingerprint=str(payload["dino_fingerprint"]),
    )

    features, metadata = cli._validate_dino_feature_cache(
        payload,
        sample_keys=sample_keys,
        expected_static_spec=expected,
    )
    assert features.dtype == torch.float32
    assert features.shape == (4, 768)
    assert metadata["feature_spec_fingerprint"] == payload[
        "feature_spec_fingerprint"
    ]
    assert len(metadata["key_order_fingerprint"]) == 64

    invalid_payloads: list[tuple[dict[str, object], str]] = []

    wrong_order = dict(payload)
    wrong_order["sample_keys"] = list(reversed(sample_keys))
    invalid_payloads.append((wrong_order, "stable catalog order"))

    wrong_catalog = dict(payload)
    wrong_catalog["catalog_fingerprint"] = "wrong"
    invalid_payloads.append((wrong_catalog, "catalog_fingerprint"))

    wrong_internal_fingerprint = dict(payload)
    wrong_internal_fingerprint["feature_spec_fingerprint"] = "wrong"
    invalid_payloads.append((wrong_internal_fingerprint, "feature_spec_fingerprint"))

    wrong_dtype = dict(payload)
    wrong_dtype["features"] = torch.as_tensor(payload["features"]).half()
    invalid_payloads.append((wrong_dtype, "dtype"))

    wrong_shape = dict(payload)
    wrong_shape["features"] = torch.as_tensor(payload["features"])[:, :-1]
    invalid_payloads.append((wrong_shape, "shape"))

    nonfinite = dict(payload)
    nonfinite_features = torch.as_tensor(payload["features"]).clone()
    nonfinite_features[0, 0] = float("nan")
    nonfinite["features"] = nonfinite_features
    invalid_payloads.append((nonfinite, "NaN or Inf"))

    zero_norm = dict(payload)
    zero_features = torch.as_tensor(payload["features"]).clone()
    zero_features[0].zero_()
    zero_norm["features"] = zero_features
    invalid_payloads.append((zero_norm, "zero-norm"))

    for invalid, match in invalid_payloads:
        with pytest.raises(ValueError, match=match):
            cli._validate_dino_feature_cache(
                invalid,
                sample_keys=sample_keys,
                expected_static_spec=expected,
            )


def test_dedup_mode_requires_a_feature_cache(
    tmp_path: Path, monkeypatch
) -> None:
    _, output_dir, _, argv, _ = _build_synthetic_inputs(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="requires --dino-feature-cache"):
        cli.main([*argv, "--dedup-mode", "dino-cosine"])
    assert not output_dir.exists()


def test_dedup_cli_dry_run_artifacts_audit_idempotence_and_conflict(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    data_root, output_dir, sample_keys, argv, _ = _build_synthetic_inputs(
        tmp_path, monkeypatch
    )
    dedup_argv, feature_path, _ = _add_dedup_inputs(
        tmp_path, argv, sample_keys
    )
    derived_path = (
        data_root / "cache" / "global_additive_dedup_scores_split0.01.pt"
    )

    assert cli.main([*dedup_argv, "--dry-run"]) == 0
    assert not output_dir.exists()
    assert not derived_path.exists()
    assert feature_path.is_file()
    dry_output = capsys.readouterr().out
    assert '"dedup_mode": "dino-cosine"' in dry_output
    assert '"uses_kmeans": false' in dry_output
    assert sample_keys[2] in dry_output

    assert cli.main(dedup_argv) == 0
    assert cli.main(dedup_argv) == 0
    for target in (2, 3, 4):
        keys_path = output_dir / f"global_additive_dedup_{target:04d}_keys.pt"
        names_path = (
            output_dir
            / f"global_additive_dedup_{target:04d}_labeled_names.txt"
        )
        keys = torch.load(keys_path, map_location="cpu", weights_only=False)
        assert len(keys) == target
        assert names_path.read_text(encoding="utf-8").splitlines() == [
            key.rsplit("/", 1)[-1] for key in keys
        ]
    assert not (output_dir / "global_additive_0002_keys.pt").exists()

    derived = torch.load(derived_path, map_location="cpu", weights_only=False)
    assert derived["dedup_mode"] == "dino-cosine"
    assert derived["dedup_threshold"] == pytest.approx(0.95)
    assert derived["dedup_comparison_rule"] == "cosine_similarity > threshold"
    assert derived["dino_feature_cache_sha256"] == cli._sha256_file(feature_path)
    assert len(derived["dino_feature_spec_fingerprint"]) == 64
    assert len(derived["dino_feature_key_order_fingerprint"]) == 64

    manifest_path = output_dir / "global_additive_dedup_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["selection"]["dedup"] is True
    assert manifest["selection"]["dedup_mode"] == "dino-cosine"
    assert manifest["selection"]["dedup_threshold"] == pytest.approx(0.95)
    assert manifest["selection"]["dedup_comparison_rule"] == (
        "cosine_similarity > threshold"
    )
    assert manifest["selection"]["dedup_rounds"][0]["skipped_count"] == 1
    assert manifest["selection"]["dedup_rounds"][1][
        "relaxed_selected_count"
    ] == 1
    assert manifest["selection"]["dedup_audit"]["relaxed_count"] == 2
    assert manifest["selection"]["dedup_audit"]["max_cosine_similarity_field"] == (
        "dedup_max_cosine_similarity"
    )
    assert manifest["dino_feature_cache"]["shape"] == [4, 768]
    assert manifest["dino_feature_cache"]["dtype"] == "float32"
    assert manifest["dino_feature_cache"]["catalog_fingerprint"] == manifest[
        "dino_feature_cache"
    ]["image_fingerprint"]

    csv_path = output_dir / "global_additive_dedup_scores.csv"
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = {row["sample_key"]: row for row in csv.DictReader(stream)}
    assert rows[sample_keys[0]]["dedup_decision"] == "seed"
    assert rows[sample_keys[1]]["dedup_decision"] == "relaxed_backfill"
    assert rows[sample_keys[1]]["dedup_max_cosine_similarity"] == "1.0000000000"
    assert rows[sample_keys[1]]["dedup_reference_key"] == sample_keys[0]
    assert rows[sample_keys[1]]["dedup_relaxed"] == "1"
    assert rows[sample_keys[2]]["dedup_decision"] == "strict_selected"
    assert rows[sample_keys[3]]["dedup_decision"] == "relaxed_backfill"

    changed_threshold_argv = list(dedup_argv)
    changed_threshold_argv[
        changed_threshold_argv.index("--dedup-threshold") + 1
    ] = "0.94"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        cli.main(changed_threshold_argv)

    changed = output_dir / "global_additive_dedup_0002_labeled_names.txt"
    changed.write_text("different\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        cli.main(dedup_argv)
