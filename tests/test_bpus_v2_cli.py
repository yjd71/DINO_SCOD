from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import select_bpus_v2
from selection.artifacts import (
    compute_catalog_fingerprint,
    file_sha256,
    load_split_keys,
    save_split_keys,
    stable_fingerprint,
)
from selection.protocol import SamplingProtocol
from utils.checkpoint_pc_hbm import (
    compute_labeled_split_fingerprint,
    state_dict_fingerprint,
)
from utils.dataloader import _load_txt_sample_keys


def test_cli_requires_explicit_counts_and_defaults_to_batch_16() -> None:
    with pytest.raises(SystemExit):
        select_bpus_v2.parse_args([])
    args = select_bpus_v2.parse_args(
        [
            "--target-counts",
            "41",
            "202",
            "404",
            "--bootstrap-split",
            "bootstrap.pt",
            "--selector-checkpoint",
            "selector_raw.pth",
            "--output-dir",
            "out",
            "--seed",
            "2025",
        ]
    )
    assert args.target_counts == [41, 202, 404]
    assert args.batch_size == 16
    assert args.formula_variant == "v2"


def test_formal_mode_rejects_other_counts_and_diagnostic_formula(
    tmp_path: Path,
) -> None:
    common = [
        "--bootstrap-split",
        str(tmp_path / "missing-bootstrap.pt"),
        "--selector-checkpoint",
        str(tmp_path / "missing-selector.pth"),
        "--output-dir",
        str(tmp_path / "output"),
        "--seed",
        "2025",
    ]
    with pytest.raises(ValueError, match="41 202 404"):
        select_bpus_v2.main(["--target-counts", "40", "200", "400", *common])
    with pytest.raises(ValueError, match="Diagnostic formula"):
        select_bpus_v2.main(
            [
                "--target-counts",
                "41",
                "202",
                "404",
                "--formula-variant",
                "v2-a",
                *common,
            ]
        )
    with pytest.raises(ValueError, match="isolated output directory"):
        select_bpus_v2.main(
            [
                "--target-counts",
                "2",
                "4",
                "6",
                "--debug-custom-counts",
                *common,
            ]
        )
    with pytest.raises(ValueError, match="isolated output directory"):
        select_bpus_v2.main(
            [
                "--target-counts",
                "41",
                "202",
                "404",
                "--debug-custom-counts",
                *common,
            ]
        )


def test_selector_identity_rejects_schema_one(tmp_path: Path) -> None:
    state = {"weight": torch.ones(1)}
    checkpoint = tmp_path / "selector_raw.pth"
    torch.save(state, checkpoint)
    config_path = tmp_path / "selector_config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "bpus_v2_selector",
                "method": "BPUS-v2",
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        selector_checkpoint=checkpoint,
        selector_config=config_path,
        seed=2025,
    )
    protocol = SamplingProtocol.from_counts((41, 202, 404))
    with pytest.raises(ValueError, match="schema_version"):
        select_bpus_v2._load_selector_identity(
            args,
            protocol=protocol,
            split_fingerprint="split",
            dino_fingerprint="dino",
            catalog_fingerprint="catalog",
        )


def test_schema_one_pool_cache_cannot_be_rebuilt_in_place(tmp_path: Path) -> None:
    cache = tmp_path / "pool_scores.pt"
    torch.save({"schema_version": 1, "method": "legacy"}, cache)
    with pytest.raises(ValueError, match="schema 2"):
        select_bpus_v2._assert_v2_cache_namespace((cache,))


def test_partial_schema_two_cache_can_only_be_recovered_by_rebuild(
    tmp_path: Path,
) -> None:
    scores = tmp_path / "pool_scores.pt"
    prototypes = tmp_path / "pool_prototypes.pt"
    torch.save({"schema_version": 2, "method": "BPUS-v2"}, scores)

    with pytest.raises(RuntimeError, match="incomplete"):
        select_bpus_v2._validate_cache_mode(
            scores,
            prototypes,
            reuse_cache=False,
            rebuild_cache=False,
            verify_only=False,
        )
    assert select_bpus_v2._validate_cache_mode(
        scores,
        prototypes,
        reuse_cache=False,
        rebuild_cache=True,
        verify_only=False,
    ) == (True, False)


def test_formal_bootstrap_quota_and_scoring_numerics_identity() -> None:
    bootstrap = [f"TR-CAMO/camo-{index}" for index in range(10)] + [
        f"TR-COD10K/cod-{index}" for index in range(31)
    ]
    assert select_bpus_v2._validate_formal_bootstrap(bootstrap) == {
        "TR-CAMO": 10,
        "TR-COD10K": 31,
    }
    with pytest.raises(ValueError, match="TR-CAMO=10"):
        select_bpus_v2._validate_formal_bootstrap(bootstrap[1:])

    base = SimpleNamespace(
        eps=1e-6,
        boundary_mass_eps=1e-6,
        device="cuda",
        amp=True,
        deterministic=True,
        batch_size=16,
        num_workers=8,
    )
    formula = select_bpus_v2._formula_spec("v2")
    select_bpus_v2._validate_formal_scoring_settings(base, formal_mode=True)
    baseline = select_bpus_v2._preprocessing_fingerprint(base, formula)
    changed = SimpleNamespace(**{**vars(base), "batch_size": 8})
    with pytest.raises(ValueError, match="batch_size=16"):
        select_bpus_v2._validate_formal_scoring_settings(
            changed, formal_mode=True
        )
    assert select_bpus_v2._preprocessing_fingerprint(changed, formula) != baseline


def test_split_txt_uses_loader_compatible_stable_keys(tmp_path: Path) -> None:
    keys = ["TR-CAMO/a", "TR-COD10K/shared-name"]
    path = tmp_path / "bpus_v2_0002_seed17.txt"
    fingerprint = select_bpus_v2._save_split_text(path, keys)

    items = [
        {"key": "TR-CAMO/a", "stem": "a"},
        {"key": "TR-COD10K/shared-name", "stem": "shared-name"},
    ]
    assert _load_txt_sample_keys(path, items) == set(keys)
    assert fingerprint == file_sha256(path)
    assert path.read_bytes() == b"TR-CAMO/a\nTR-COD10K/shared-name\n"

    with pytest.raises(FileExistsError, match="different TXT split"):
        select_bpus_v2._save_split_text(path, ["TR-CAMO/different"])
    select_bpus_v2._save_split_text(
        path,
        ["TR-CAMO/different"],
        refuse_mismatch=False,
    )
    assert path.read_bytes() == b"TR-CAMO/different\n"


def test_score_pool_never_enables_amp_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TinyPool(torch.utils.data.Dataset):
        sample_keys = ["TR-CAMO/a"]

        def __len__(self) -> int:
            return 1

        def __getitem__(self, _index: int):
            return self.sample_keys[0], torch.zeros(3, 392, 392)

    class TinyDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.pc_hbm = None

    class TinyBaseModel(torch.nn.Module):
        def __init__(self, pc_cfg=None) -> None:
            super().__init__()
            assert pc_cfg is None
            self.decoder = TinyDecoder()
            self.dino = torch.nn.Linear(1, 1)

    calls: list[dict[str, object]] = []

    def fake_score(_model, images, **kwargs):
        calls.append(kwargs)
        batch = images.shape[0]
        return SimpleNamespace(
            boundary_disagreement=torch.full((batch,), 0.5),
            global_disagreement=torch.full((batch,), 0.2),
            boundary_value=torch.full((batch,), 0.4),
            boundary_mass=torch.ones(batch),
            valid_boundary=torch.ones(batch, dtype=torch.bool),
            prototype=torch.nn.functional.normalize(
                torch.ones(batch, 128), dim=1
            ),
        )

    monkeypatch.setattr(select_bpus_v2, "BaseModel", TinyBaseModel)
    monkeypatch.setattr(select_bpus_v2, "score_and_prototype_bpus_v2", fake_score)
    args = SimpleNamespace(
        device="cpu",
        batch_size=1,
        num_workers=0,
        amp=True,
        eps=1e-6,
        boundary_mass_eps=1e-6,
    )
    scores, prototypes = select_bpus_v2._score_pool(
        args,
        pool=TinyPool(),
        state={"weight": torch.zeros(1)},
        value_mode="smooth-value",
    )
    assert calls == [
        {
            "eps": 1e-6,
            "boundary_mass_eps": 1e-6,
            "use_amp": False,
            "value_mode": "smooth-value",
        }
    ]
    assert scores["valid_boundary"].tolist() == [True]
    assert prototypes["prototypes"].shape == (1, 128)


def test_synthetic_schema_two_round_trip_verify_and_tamper_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_keys = [
        "TR-CAMO/a",
        "TR-CAMO/b",
        "TR-COD10K/c",
        "TR-COD10K/d",
        "TR-COD10K/e",
        "TR-COD10K/f",
    ]
    image_paths: list[str] = []
    for index, key in enumerate(sample_keys):
        image = tmp_path / f"image_{index}.jpg"
        image.write_bytes(f"rgb-{key}".encode("utf-8"))
        image_paths.append(str(image))

    class TinyPool:
        def __init__(self, *_args, **_kwargs) -> None:
            self.sample_keys = list(sample_keys)
            self.items = [
                {"key": key, "image": image}
                for key, image in zip(sample_keys, image_paths)
            ]

        def __len__(self) -> int:
            return len(self.sample_keys)

    bootstrap = sample_keys[:2]
    bootstrap_path = tmp_path / "bootstrap.pt"
    save_split_keys(bootstrap_path, bootstrap)
    dino_path = tmp_path / "dino.pth"
    dino_path.write_bytes(b"dino-v2")
    state = {"weight": torch.tensor([1.0])}
    selector_path = tmp_path / "selector_raw.pth"
    torch.save(state, selector_path)
    protocol = SamplingProtocol.from_counts((2, 4, 6), allow_custom=True)
    training_config = {"epochs": 2, "batch_size": 1}
    selector_config = {
        "schema_version": 2,
        "kind": "bpus_v2_selector",
        "method": "BPUS-v2",
        "formal_mode": False,
        "debug_mode": True,
        "protocol": protocol.name,
        "target_counts": list(protocol.target_counts),
        "bootstrap_count": protocol.bootstrap_count,
        "seed": 17,
        "catalog_count": len(sample_keys),
        "catalog_fingerprint": compute_catalog_fingerprint(sample_keys),
        "split_fingerprint": compute_labeled_split_fingerprint(bootstrap),
        "bootstrap_fingerprint": compute_labeled_split_fingerprint(bootstrap),
        "dino_weight_fingerprint": file_sha256(dino_path),
        "selector_fingerprint": state_dict_fingerprint(state),
        "training_config": training_config,
        "training_fingerprint": stable_fingerprint(training_config),
        "epochs": 2,
    }
    config_path = tmp_path / "selector_config.json"
    config_path.write_text(json.dumps(selector_config), encoding="utf-8")
    output_dir = tmp_path / "debug_bpus_v2"

    prototypes = torch.zeros(6, 128)
    for index in range(6):
        prototypes[index, index] = 1.0

    def fake_score_pool(*_args, **_kwargs):
        return (
            {
                "boundary_disagreement": torch.tensor(
                    [0.2, 0.3, 0.8, 0.7, 0.6, 0.5]
                ),
                "global_disagreement": torch.tensor(
                    [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
                ),
                "boundary_value": torch.tensor(
                    [0.18, 0.27, 0.72, 0.63, 0.54, 0.45]
                ),
                "boundary_mass": torch.ones(6),
                "valid_boundary": torch.ones(6, dtype=torch.bool),
            },
            {
                "prototypes": prototypes,
                "valid_boundary": torch.ones(6, dtype=torch.bool),
            },
        )

    monkeypatch.setattr(select_bpus_v2, "DINO_WEIGHT_PATH", dino_path)
    monkeypatch.setattr(select_bpus_v2, "SelectionPoolDataset", TinyPool)
    monkeypatch.setattr(select_bpus_v2, "_score_pool", fake_score_pool)
    common = [
        "--target-counts",
        "2",
        "4",
        "6",
        "--debug-custom-counts",
        "--data-root",
        str(tmp_path),
        "--bootstrap-split",
        str(bootstrap_path),
        "--selector-checkpoint",
        str(selector_path),
        "--selector-config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--seed",
        "17",
        "--device",
        "cpu",
        "--num-workers",
        "0",
        "--no-amp",
    ]
    assert select_bpus_v2.main(common) == 0

    score_payload = torch.load(
        output_dir / "pool_scores.pt", map_location="cpu", weights_only=False
    )
    prototype_payload = torch.load(
        output_dir / "pool_prototypes.pt", map_location="cpu", weights_only=False
    )
    acquisition = torch.load(
        output_dir / "acquisition_order.pt", map_location="cpu", weights_only=False
    )
    assert score_payload["schema_version"] == 2
    assert set(
        (
            "boundary_disagreement",
            "global_disagreement",
            "boundary_value",
            "boundary_mass",
            "valid_boundary",
        )
    ).issubset(score_payload)
    assert prototype_payload["prototype_level"] == "p2"
    assert prototype_payload["prototype_dim"] == 128
    assert len(acquisition["acquired_keys"]) == 4
    assert acquisition["cumulative_subset_counts"]["TR-CAMO"].numel() == 4
    assert load_split_keys(output_dir / "bpus_v2_0002_seed17.pt") == bootstrap
    assert set(load_split_keys(output_dir / "bpus_v2_0002_seed17.pt")) < set(
        load_split_keys(output_dir / "bpus_v2_0004_seed17.pt")
    ) < set(load_split_keys(output_dir / "bpus_v2_0006_seed17.pt"))

    for count in protocol.target_counts:
        pt_path = output_dir / f"bpus_v2_{count:04d}_seed17.pt"
        txt_path = pt_path.with_suffix(".txt")
        pt_keys = load_split_keys(pt_path)
        assert txt_path.read_text(encoding="utf-8").splitlines() == pt_keys
        assert txt_path.read_bytes() == (
            "\n".join(pt_keys) + "\n"
        ).encode("utf-8")

    manifest_path = output_dir / "selection_manifest.json"
    manifest_before = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_before["files"]["split_txt"] == {
        str(count): f"bpus_v2_{count:04d}_seed17.txt"
        for count in protocol.target_counts
    }
    assert manifest_before["split_txt_fingerprints"] == {
        str(count): file_sha256(
            output_dir / f"bpus_v2_{count:04d}_seed17.txt"
        )
        for count in protocol.target_counts
    }
    for count in protocol.target_counts:
        pt_keys = load_split_keys(output_dir / f"bpus_v2_{count:04d}_seed17.pt")
        assert manifest_before["split_fingerprints"][str(count)] == (
            compute_labeled_split_fingerprint(pt_keys)
        )
    runtime_path = output_dir / "runtime_report.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["elapsed_seconds"] = 999.0
    runtime_path.write_text(json.dumps(runtime), encoding="utf-8")

    monkeypatch.setattr(
        select_bpus_v2,
        "_score_pool",
        lambda *_args, **_kwargs: pytest.fail("verify-only must not rescore"),
    )
    assert select_bpus_v2.main([*common, "--verify-only"]) == 0
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest_before

    txt_path = output_dir / "bpus_v2_0002_seed17.txt"
    canonical_txt = txt_path.read_bytes()
    txt_path.write_text("TR-CAMO/not-the-pt-split\n", encoding="utf-8")
    with pytest.raises(ValueError, match="TXT split disagrees with its PT peer"):
        select_bpus_v2.main([*common, "--verify-only"])
    txt_path.write_bytes(canonical_txt)
    assert select_bpus_v2.main([*common, "--verify-only"]) == 0

    acquisition["utility"][0] += 1.0
    torch.save(acquisition, output_dir / "acquisition_order.pt")
    with pytest.raises(ValueError, match="deterministic replay"):
        select_bpus_v2.main([*common, "--verify-only"])
