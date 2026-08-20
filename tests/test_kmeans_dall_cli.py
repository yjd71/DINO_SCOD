from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

import select_kmeans_dall as cli
from configs.pc_hbm_dino_config import DinoPCHBMConfig
from utils.checkpoint_pc_hbm import (
    build_artifact_metadata,
    compute_labeled_split_fingerprint,
    save_decoder_checkpoint,
)
from utils.dataloader import SelectionPoolDataset
from utils.kmeans_only import fit_kmeans_only


class _TinyLegacyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(2, 1, kernel_size=1, bias=True)


def _build_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path, list[str], list[str], dict[str, object]]:
    data_root = tmp_path / "Dataset" / "COD"
    for subset, names in (
        ("SET-A", ("a0", "a1", "a2")),
        ("SET-B", ("b0", "b1", "b2")),
    ):
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
    full_feature_spec = {
        **static_spec,
        "feature_amp": False,
        "feature_device_type": "cpu",
    }
    features = torch.zeros(6, 768, dtype=torch.float32)
    features[0, 0] = 1.0
    features[1, 0], features[1, 2] = 0.7, 0.714
    features[2, 0], features[2, 2] = 0.7, -0.714
    features[3, 1] = 1.0
    features[4, 1], features[4, 3] = 0.7, 0.714
    features[5, 1], features[5, 3] = 0.7, -0.714
    feature_payload = {
        **full_feature_spec,
        "feature_spec_fingerprint": cli._sha256_json(full_feature_spec),
        "normalized": False,
        "sample_keys": sample_keys,
        "features": features,
    }
    feature_path = data_root / "cache" / "features.pt"
    feature_path.parent.mkdir(parents=True)
    torch.save(feature_payload, feature_path)

    kmeans_fit = fit_kmeans_only(
        sample_keys,
        features,
        n_clusters=2,
        random_seed=2025,
    )
    seed_keys = sorted(kmeans_fit.seed_keys)
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
    _, _, selector_fingerprint = cli._load_selector_identity(
        checkpoint_path,
        selector_seed_keys=seed_keys,
        dino_fingerprint=dino_fingerprint,
    )
    score_spec = cli._expected_source_spec(
        catalog_fingerprint=catalog_fingerprint,
        selector_fingerprint=selector_fingerprint,
        dino_fingerprint=dino_fingerprint,
    )
    boundary = torch.tensor([0.0, 0.8, 0.4, 0.0, 0.7, 0.3], dtype=torch.float32)
    global_disagreement = torch.tensor(
        [0.5, 0.95, 0.05, 0.5, 0.9, 0.1], dtype=torch.float32
    )
    score_payload: dict[str, object] = {
        **score_spec,
        "sample_keys": sample_keys,
        "boundary_disagreement": boundary,
        "global_disagreement": global_disagreement,
        "scores": boundary * (1.0 - global_disagreement),
    }
    score_path = data_root / "cache" / "scores.pt"
    torch.save(score_payload, score_path)

    output_root = data_root / "splits" / "kmeans_dall_dedup"
    argv = [
        "--data-root",
        str(data_root),
        "--train-sets",
        "SET-A",
        "SET-B",
        "--dino-feature-cache",
        str(feature_path),
        "--selector-checkpoint",
        str(checkpoint_path),
        "--source-score-cache",
        str(score_path),
        "--directions",
        "high",
        "low",
        "--n-clusters",
        "2",
        "--seed",
        "2025",
        "--target-counts",
        "3",
        "4",
        "6",
        "--dedup-threshold",
        "0.95",
        "--output-root",
        str(output_root),
    ]
    return data_root, output_root, sample_keys, argv, score_payload


def test_cli_defaults_match_formal_dual_direction_protocol() -> None:
    args = cli.parse_args(
        [
            "--selector-checkpoint",
            "selector.pth",
            "--source-score-cache",
            "scores.pt",
        ]
    )
    assert args.directions == ["high", "low"]
    assert args.n_clusters == 40
    assert args.seed == 2025
    assert args.target_counts == [41, 202, 404, 808]
    assert args.dedup_threshold == pytest.approx(0.95)
    assert args.output_root.as_posix().endswith(
        "Dataset/COD/splits/kmeans_dall_dedup"
    )


def test_cli_source_has_no_model_forward_or_selection_use_of_other_scores() -> None:
    source = Path(cli.__file__).read_text(encoding="utf-8")
    algorithm = (Path(cli.__file__).parent / "utils" / "kmeans_dall.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "BaseModel",
        "score_pool(",
        "compute_pc_bacs_score",
        "SAMLabel",
        "pseudo_label",
    ):
        assert forbidden not in source + algorithm
    assert "boundary_disagreement" not in algorithm
    assert "scores" not in algorithm
    assert "1.0 - disagreement" not in algorithm


def test_dry_run_dual_outputs_idempotence_and_conflict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, output_root, sample_keys, argv, _ = _build_inputs(tmp_path, monkeypatch)

    assert cli.main([*argv, "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert set(report["directions"]) == {"high", "low"}
    assert report["ranking_uses_global_disagreement"] is True
    assert report["ranking_uses_boundary_disagreement"] is False
    assert report["ranking_uses_pc_bacs_value"] is False
    assert report["model_forward"] is False
    assert report["directions"]["high"]["targets"] != report["directions"]["low"]["targets"]
    assert not output_root.exists()

    assert cli.main(argv) == 0
    capsys.readouterr()
    assert cli.main(argv) == 0
    capsys.readouterr()

    for direction in ("high", "low"):
        direction_dir = output_root / direction
        previous: set[str] = set()
        prefix = f"kmeans_dall_{direction}_dedup"
        for label in (2, 3, 4, 6):
            artifact_stem = f"{prefix}_{label:04d}"
            if label == 2:
                artifact_stem += "_seed"
            pt_path = direction_dir / f"{artifact_stem}_keys.pt"
            txt_path = direction_dir / f"{artifact_stem}_labeled_names.txt"
            keys = torch.load(pt_path, map_location="cpu", weights_only=False)
            names = txt_path.read_text(encoding="utf-8").splitlines()
            assert isinstance(keys, list)
            assert keys == sorted(keys)
            assert len(keys) == len(set(keys)) == label
            assert names == [key.rsplit("/", 1)[-1] for key in keys]
            assert previous <= set(keys)
            previous = set(keys)

        with (direction_dir / f"{prefix}_assignments.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == len(sample_keys)
        assert {row["dall_direction"] for row in rows} == {direction}
        assert sum(int(row["is_center_seed"]) for row in rows) == 2
        assert sum(int(row["selected_0003"]) for row in rows) == 3
        assert sum(int(row["selected_0006"]) for row in rows) == 6

        manifest = json.loads(
            (direction_dir / f"{prefix}_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["protocol"] == "kmeans_dall_incluster_dino_dedup_v1"
        assert manifest["direction"] == direction
        assert manifest["selection"]["uses_global_disagreement"] is True
        assert manifest["selection"]["uses_boundary_disagreement"] is False
        assert manifest["selection"]["uses_pc_bacs_value"] is False
        assert manifest["selection"]["uses_one_minus_global_disagreement"] is False

    changed = output_root / "high" / "kmeans_dall_high_dedup_0003_labeled_names.txt"
    changed.write_text("different\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        cli.main(argv)


def test_changing_boundary_and_product_does_not_change_dry_run_splits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, _, _, argv, payload = _build_inputs(tmp_path, monkeypatch)
    assert cli.main([*argv, "--dry-run"]) == 0
    first = json.loads(capsys.readouterr().out)

    global_disagreement = torch.as_tensor(payload["global_disagreement"])
    replacement_boundary = torch.full_like(global_disagreement, 0.25)
    payload["boundary_disagreement"] = replacement_boundary
    payload["scores"] = replacement_boundary * (1.0 - global_disagreement)
    score_path = Path(argv[argv.index("--source-score-cache") + 1])
    torch.save(payload, score_path)
    assert cli.main([*argv, "--dry-run"]) == 0
    second = json.loads(capsys.readouterr().out)

    assert first["directions"] == second["directions"]


@pytest.mark.parametrize(
    ("cache_kind", "field", "replacement", "match"),
    [
        ("feature", "catalog_fingerprint", "wrong", "catalog_fingerprint"),
        ("feature", "features", "half", "dtype"),
        ("score", "catalog_fingerprint", "wrong", "catalog_fingerprint"),
        ("score", "global_disagreement", "half", "dtype"),
    ],
)
def test_cache_mismatches_are_hard_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache_kind: str,
    field: str,
    replacement: str,
    match: str,
) -> None:
    _, output_root, _, argv, _ = _build_inputs(tmp_path, monkeypatch)
    flag = "--dino-feature-cache" if cache_kind == "feature" else "--source-score-cache"
    path = Path(argv[argv.index(flag) + 1])
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if replacement == "half":
        payload[field] = payload[field].half()
    else:
        payload[field] = replacement
    torch.save(payload, path)

    with pytest.raises(ValueError, match=match):
        cli.main(argv)
    assert not output_root.exists()


def test_selector_seed_identity_and_duplicate_directions_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, output_root, _, argv, _ = _build_inputs(tmp_path, monkeypatch)
    checkpoint_path = Path(argv[argv.index("--selector-checkpoint") + 1])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["artifact_meta"]["labeled_split_fingerprint"] = "wrong"
    torch.save(payload, checkpoint_path)
    with pytest.raises(ValueError, match="labeled_split_fingerprint"):
        cli.main(argv)
    assert not output_root.exists()

    duplicate_argv = list(argv)
    direction_index = duplicate_argv.index("--directions")
    duplicate_argv[direction_index + 1 : direction_index + 3] = ["high", "high"]
    with pytest.raises(ValueError, match="unique"):
        cli.main(duplicate_argv)
