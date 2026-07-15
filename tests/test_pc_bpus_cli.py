from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

import select_pc_bpus as select_cli
import train_selector_pc_bpus as train_cli
from selection.artifacts import (
    compute_catalog_fingerprint,
    file_sha256,
    load_split_keys,
    stable_fingerprint,
)
from selection.protocol import SamplingProtocol
from utils.checkpoint_pc_hbm import (
    compute_labeled_split_fingerprint,
    state_dict_fingerprint,
)


def test_both_cli_entry_points_require_explicit_target_counts() -> None:
    with pytest.raises(SystemExit):
        select_cli.parse_args(
            [
                "--bootstrap-split",
                "bootstrap.pt",
                "--selector-checkpoint",
                "selector.pth",
                "--output-dir",
                "out",
                "--seed",
                "2025",
            ]
        )
    with pytest.raises(SystemExit):
        train_cli.parse_args(
            [
                "--labeled-indices-pt",
                "bootstrap.pt",
                "--output-dir",
                "out",
                "--seed",
                "2025",
            ]
        )


def test_selector_and_pool_use_the_same_im_directory_layout(tmp_path: Path) -> None:
    args = SimpleNamespace(data_root=tmp_path, train_sets=["TR-CAMO", "TR-COD10K"])
    expected = [
        str(tmp_path / "TR-CAMO" / "im"),
        str(tmp_path / "TR-COD10K" / "im"),
    ]
    assert select_cli._image_roots(args) == expected
    assert train_cli._image_roots(args) == expected


def test_formal_selector_rejects_non_4040_catalog_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dino_weight = tmp_path / "dino.pth"
    dino_weight.write_bytes(b"fake-dino")
    bootstrap = tmp_path / "bootstrap.pt"
    torch.save(["TR-CAMO/a"], bootstrap)

    class TinyPool:
        sample_keys = ["TR-CAMO/a"]

    monkeypatch.setattr(train_cli, "DINO_WEIGHT_PATH", dino_weight)
    monkeypatch.setattr(train_cli, "SelectionPoolDataset", lambda *_args, **_kwargs: TinyPool())

    with pytest.raises(RuntimeError, match="requires 4040 images"):
        train_cli.main(
            [
                "--data-root",
                str(tmp_path),
                "--target-counts",
                "41",
                "202",
                "404",
                "--labeled-indices-pt",
                str(bootstrap),
                "--output-dir",
                str(tmp_path / "selector"),
                "--seed",
                "2025",
                "--device",
                "cpu",
                "--dry-run",
            ]
        )


def test_score_pool_never_enables_amp_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    class TinyPool(Dataset):
        sample_keys = ["TR-CAMO/a"]

        def __len__(self) -> int:
            return 1

        def __getitem__(self, index: int):
            assert index == 0
            return self.sample_keys[index], torch.zeros(3, 4, 4)

    class TinyBaseModel(nn.Module):
        def __init__(self, pc_cfg=None) -> None:
            super().__init__()
            assert pc_cfg is None
            self.dino = nn.Identity()
            self.decoder = nn.Linear(1, 1, bias=False)

    observed: list[bool] = []

    def fake_score(_model, images, **kwargs):
        observed.append(bool(kwargs["use_amp"]))
        batch = images.shape[0]
        return SimpleNamespace(
            d_boundary=torch.full((batch,), 0.5),
            d_global=torch.full((batch,), 0.1),
            value=torch.full((batch,), 0.2),
            boundary_mass=torch.ones(batch),
            valid=torch.ones(batch, dtype=torch.bool),
            prototype=torch.nn.functional.one_hot(
                torch.zeros(batch, dtype=torch.long), num_classes=128
            ).float(),
        )

    monkeypatch.setattr(select_cli, "BaseModel", TinyBaseModel)
    monkeypatch.setattr(select_cli, "score_and_prototype", fake_score)
    args = SimpleNamespace(
        device="cpu",
        batch_size=1,
        num_workers=0,
        eps=1.0e-6,
        boundary_mass_eps=1.0e-6,
        amp=True,
    )
    select_cli._score_pool(
        args,
        pool=TinyPool(),
        state={"weight": torch.ones(1, 1)},
    )
    assert observed == [False]


def test_synthetic_pool_cache_nested_splits_and_strict_verify_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys = [
        "TR-CAMO/a",
        "TR-CAMO/b",
        "TR-COD10K/c",
        "TR-COD10K/d",
        "TR-COD10K/e",
    ]
    image_paths = []
    for index, key in enumerate(keys):
        image_path = tmp_path / f"image_{index}.jpg"
        image_path.write_bytes(f"rgb-{key}".encode("utf-8"))
        image_paths.append(str(image_path))

    class TinyPool:
        sample_keys = keys
        items = [
            {"key": key, "image": image_path}
            for key, image_path in zip(keys, image_paths)
        ]

        def __len__(self) -> int:
            return len(self.sample_keys)

    dino_weight = tmp_path / "dino.pth"
    dino_weight.write_bytes(b"fake-dino-weight")
    bootstrap_keys = keys[:2]
    bootstrap_path = tmp_path / "bootstrap.pt"
    torch.save(bootstrap_keys, bootstrap_path)
    state = {"weight": torch.tensor([1.0])}
    checkpoint = tmp_path / "selector_raw.pth"
    torch.save(state, checkpoint)
    protocol = SamplingProtocol.from_counts((2, 3, 4), allow_custom=True)
    selector_config = {
        "kind": "pc_bpus_selector",
        "protocol": protocol.name,
        "target_counts": list(protocol.target_counts),
        "seed": 7,
        "split_fingerprint": compute_labeled_split_fingerprint(bootstrap_keys),
        "dino_weight_fingerprint": file_sha256(dino_weight),
        "catalog_fingerprint": compute_catalog_fingerprint(keys),
        "selector_fingerprint": state_dict_fingerprint(state),
    }
    config_path = tmp_path / "selector_config.json"
    config_path.write_text(json.dumps(selector_config), encoding="utf-8")

    prototypes = torch.zeros(len(keys), 128)
    prototypes[torch.arange(len(keys)), torch.arange(len(keys))] = 1.0
    values = torch.tensor([0.05, 0.05, 0.60, 0.90, 0.70])

    def fake_score_pool(*_args, **_kwargs):
        return (
            {
                "d_boundary": torch.full((len(keys),), 0.8),
                "d_global": torch.full((len(keys),), 0.1),
                "value": values,
                "boundary_mass": torch.ones(len(keys)),
                "valid_boundary": torch.ones(len(keys), dtype=torch.bool),
            },
            {
                "prototypes": prototypes,
                "valid_boundary": torch.ones(len(keys), dtype=torch.bool),
            },
        )

    monkeypatch.setattr(select_cli, "DINO_WEIGHT_PATH", dino_weight)
    monkeypatch.setattr(select_cli, "SelectionPoolDataset", lambda *_args, **_kwargs: TinyPool())
    monkeypatch.setattr(select_cli, "_score_pool", fake_score_pool)
    monkeypatch.setattr(select_cli, "_set_seed", lambda *_args, **_kwargs: None)

    output_dir = tmp_path / "debug_pc_bpus"
    common_args = [
        "--data-root",
        str(tmp_path),
        "--target-counts",
        "2",
        "3",
        "4",
        "--debug-custom-counts",
        "--bootstrap-split",
        str(bootstrap_path),
        "--selector-checkpoint",
        str(checkpoint),
        "--selector-config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--seed",
        "7",
        "--device",
        "cpu",
        "--num-workers",
        "0",
    ]
    assert select_cli.main(common_args) == 0

    splits = {
        count: load_split_keys(
            output_dir / f"pc_bpus_{count:04d}_seed7.pt", expected_count=count
        )
        for count in (2, 3, 4)
    }
    assert splits[2] == bootstrap_keys
    assert set(splits[2]) < set(splits[3]) < set(splits[4])

    acquisition = torch.load(
        output_dir / "acquisition_order.pt", map_location="cpu", weights_only=False
    )
    acquisition_base = {
        key: value
        for key, value in acquisition.items()
        if key != "acquisition_fingerprint"
    }
    assert acquisition["acquisition_fingerprint"] == stable_fingerprint(acquisition_base)
    manifest = json.loads((output_dir / "selection_manifest.json").read_text("utf-8"))
    manifest_base = {
        key: value for key, value in manifest.items() if key != "manifest_fingerprint"
    }
    assert manifest["manifest_fingerprint"] == stable_fingerprint(manifest_base)

    assert select_cli.main([*common_args, "--verify-only"]) == 0

    acquisition["seed"] = 8
    torch.save(acquisition, output_dir / "acquisition_order.pt")
    with pytest.raises(ValueError, match="acquisition_order.pt .*disagree"):
        select_cli.main([*common_args, "--verify-only"])
