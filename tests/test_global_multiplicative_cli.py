from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from torch import nn

import select_global_multiplicative as cli
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
    monkeypatch.setattr(cli, "EXPECTED_CATALOG_COUNT", len(sample_keys))
    monkeypatch.setattr(cli, "EXPECTED_SEED_COUNT", len(seed_keys))
    monkeypatch.setattr(
        cli,
        "EXPECTED_KMEANS_SEED_FINGERPRINT",
        compute_labeled_split_fingerprint(seed_keys),
    )

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
    return output_dir, sample_keys, argv, source_payload


def test_cli_source_does_not_import_or_call_kmeans_or_model() -> None:
    source = (
        Path(__file__).parents[1] / "select_global_multiplicative.py"
    ).read_text(encoding="utf-8")
    assert "sklearn" not in source
    assert "KMeans(" not in source
    assert "fit_dino_kmeans" not in source
    assert "BaseModel" not in source
    assert "dino-feature-cache" not in source


def test_dry_run_writes_nothing_and_formal_run_is_idempotent(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    output_dir, sample_keys, argv, _ = _build_synthetic_inputs(tmp_path, monkeypatch)

    assert cli.main([*argv, "--dry-run"]) == 0
    assert not output_dir.exists()
    dry_output = capsys.readouterr().out
    assert '"uses_kmeans_during_selection": false' in dry_output
    assert '"dedup": false' in dry_output
    assert sample_keys[1] in dry_output

    assert cli.main(argv) == 0
    assert cli.main(argv) == 0
    for target in (2, 3, 4):
        pt_path = output_dir / f"global_multiplicative_{target:04d}_keys.pt"
        txt_path = (
            output_dir
            / f"global_multiplicative_{target:04d}_labeled_names.txt"
        )
        keys = torch.load(pt_path, map_location="cpu", weights_only=False)
        names = txt_path.read_text(encoding="utf-8").splitlines()
        assert len(keys) == target
        assert names == [key.rsplit("/", 1)[-1] for key in keys]

    manifest = json.loads(
        (output_dir / "global_multiplicative_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["protocol"] == cli.GLOBAL_MULTIPLICATIVE_PROTOCOL_VERSION
    assert manifest["selection"]["uses_kmeans_during_selection"] is False
    assert manifest["selection"]["seed_origin"] == "kmeans_center_40"
    assert manifest["selection"]["dedup"] is False
    assert manifest["source_cache"]["exact_source_scores_match"] is True

    with (output_dir / "global_multiplicative_scores.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        rows = {row["sample_key"]: row for row in csv.DictReader(stream)}
    assert rows[sample_keys[0]]["is_seed"] == "1"
    assert rows[sample_keys[1]]["global_candidate_rank"] == "1"
    assert rows[sample_keys[1]]["global_multiplicative_score"] == "0.7199999690"

    changed = output_dir / "global_multiplicative_0002_labeled_names.txt"
    changed.write_text("different\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        cli.main(argv)


def test_source_cache_identity_and_exact_score_mismatches_are_hard_failures(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir, _, argv, payload = _build_synthetic_inputs(tmp_path, monkeypatch)
    source_path = Path(argv[argv.index("--source-score-cache") + 1])

    wrong_catalog = dict(payload)
    wrong_catalog["catalog_fingerprint"] = "wrong-catalog"
    torch.save(wrong_catalog, source_path)
    with pytest.raises(ValueError, match="catalog_fingerprint"):
        cli.main(argv)
    assert not output_dir.exists()

    almost_equal = dict(payload)
    almost_equal["scores"] = payload["scores"].clone()
    almost_equal["scores"][3] = torch.nextafter(
        almost_equal["scores"][3], torch.tensor(float("inf"))
    )
    torch.save(almost_equal, source_path)
    with pytest.raises(ValueError, match="exact float32"):
        cli.main(argv)
    assert not output_dir.exists()


def test_selector_identity_and_fixed_seed_are_strictly_validated(
    tmp_path: Path, monkeypatch
) -> None:
    output_dir, sample_keys, argv, _ = _build_synthetic_inputs(tmp_path, monkeypatch)
    checkpoint_path = Path(argv[argv.index("--selector-checkpoint") + 1])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint["artifact_meta"]["pc_frozen"] = False
    torch.save(checkpoint, checkpoint_path)
    with pytest.raises(ValueError, match="pc_frozen"):
        cli.main(argv)
    assert not output_dir.exists()

    checkpoint["artifact_meta"]["pc_frozen"] = True
    torch.save(checkpoint, checkpoint_path)
    seed_path = Path(argv[argv.index("--seed-split") + 1])
    torch.save([sample_keys[1]], seed_path)
    with pytest.raises(ValueError, match="fixed KMeans-center"):
        cli.main(argv)
    assert not output_dir.exists()
