from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import train_selector_bpus_v2 as trainer


def _mode_args(
    tmp_path: Path,
    *,
    counts: tuple[int, int, int] = (41, 202, 404),
    epochs: int = 30,
    debug: bool = False,
    dirname: str = "selector",
) -> SimpleNamespace:
    return SimpleNamespace(
        target_counts=list(counts),
        epochs=epochs,
        debug=debug,
        output_dir=tmp_path / dirname,
    )


def test_trainer_requires_explicit_target_counts_and_epochs() -> None:
    common = [
        "--labeled-indices-pt",
        "bootstrap.pt",
        "--output-dir",
        "selector",
        "--seed",
        "2025",
    ]
    with pytest.raises(SystemExit):
        trainer.parse_args([*common, "--epochs", "30"])
    with pytest.raises(SystemExit):
        trainer.parse_args(
            [*common, "--target-counts", "41", "202", "404"]
        )

    parsed = trainer.parse_args(
        [
            *common,
            "--target-counts",
            "41",
            "202",
            "404",
            "--epochs",
            "30",
        ]
    )
    assert parsed.num_workers == 0


def test_formal_protocol_is_exactly_scout_counts_and_30_epochs(
    tmp_path: Path,
) -> None:
    protocol, formal = trainer._validate_protocol_and_mode(_mode_args(tmp_path))
    assert protocol.target_counts == (41, 202, 404)
    assert formal is True

    with pytest.raises(ValueError, match="requires --target-counts 41 202 404"):
        trainer._validate_protocol_and_mode(
            _mode_args(tmp_path, counts=(40, 200, 400))
        )
    with pytest.raises(ValueError, match="requires --target-counts 41 202 404"):
        trainer._validate_protocol_and_mode(_mode_args(tmp_path, epochs=15))


def test_debug_deviations_require_isolated_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="containing 'debug'"):
        trainer._validate_protocol_and_mode(
            _mode_args(tmp_path, epochs=2, debug=True)
        )

    protocol, formal = trainer._validate_protocol_and_mode(
        _mode_args(
            tmp_path,
            counts=(2, 3, 4),
            epochs=2,
            debug=True,
            dirname="debug_selector",
        )
    )
    assert protocol.target_counts == (2, 3, 4)
    assert formal is False


def test_formal_training_hyperparameters_are_fixed() -> None:
    args = SimpleNamespace(
        batch_size=8,
        learning_rate=1.0e-4,
        min_learning_rate=1.0e-7,
        amp=True,
        deterministic=True,
        num_workers=0,
        train_sets=["TR-CAMO", "TR-COD10K"],
        device="cuda",
        dry_run=False,
    )
    trainer._validate_formal_training_settings(args, formal_mode=True)
    args.batch_size = 4
    with pytest.raises(ValueError, match="--batch-size=8"):
        trainer._validate_formal_training_settings(args, formal_mode=True)
    trainer._validate_formal_training_settings(args, formal_mode=False)


def test_formal_training_rejects_workers_and_train_set_drift() -> None:
    args = SimpleNamespace(
        batch_size=8,
        learning_rate=1.0e-4,
        min_learning_rate=1.0e-7,
        amp=True,
        deterministic=True,
        num_workers=1,
        train_sets=["TR-CAMO", "TR-COD10K"],
        device="cuda",
        dry_run=False,
    )
    with pytest.raises(ValueError, match="--num-workers=0"):
        trainer._validate_formal_training_settings(args, formal_mode=True)

    args.num_workers = 0
    args.train_sets = ["TR-COD10K", "TR-CAMO"]
    with pytest.raises(ValueError, match="--train-sets TR-CAMO TR-COD10K"):
        trainer._validate_formal_training_settings(args, formal_mode=True)


def test_formal_bootstrap_requires_scout_prefix_quotas() -> None:
    bootstrap = [f"TR-CAMO/camo_{index:02d}" for index in range(10)]
    bootstrap.extend(
        f"TR-COD10K/cod_{index:02d}" for index in range(31)
    )
    assert trainer._validate_formal_bootstrap(bootstrap) == {
        "TR-CAMO": 10,
        "TR-COD10K": 31,
    }

    invalid = [*bootstrap[:-1], "TR-CAMO/extra"]
    with pytest.raises(ValueError, match="TR-CAMO=10 and TR-COD10K=31"):
        trainer._validate_formal_bootstrap(invalid)


def test_resume_identity_rejects_v1_payload() -> None:
    expected = trainer._resume_identity(
        seed=2025,
        target_counts=(41, 202, 404),
        split_fingerprint="bootstrap",
        dino_fingerprint="dino",
        catalog_fingerprint="catalog",
        training_fingerprint="training",
        formal_mode=True,
        debug_mode=False,
    )
    legacy_payload = {
        **expected,
        "schema_version": 1,
        "kind": "pc_bpus_selector_resume",
        "method": "PC-BPUS",
    }
    with pytest.raises(ValueError, match="schema_version mismatch"):
        trainer._validate_resume_identity(legacy_payload, expected)


def test_resume_identity_binds_training_and_data_fingerprints() -> None:
    expected = trainer._resume_identity(
        seed=2026,
        target_counts=(41, 202, 404),
        split_fingerprint="bootstrap",
        dino_fingerprint="dino",
        catalog_fingerprint="catalog",
        training_fingerprint="training",
        formal_mode=True,
        debug_mode=False,
    )
    trainer._validate_resume_identity(expected, expected)
    for field in (
        "seed",
        "target_counts",
        "bootstrap_fingerprint",
        "dino_weight_fingerprint",
        "catalog_fingerprint",
        "training_fingerprint",
        "formal_mode",
        "debug_mode",
    ):
        tampered = dict(expected)
        tampered[field] = "tampered"
        with pytest.raises(ValueError, match=field):
            trainer._validate_resume_identity(tampered, expected)


def test_resume_identity_records_exactly_one_mode() -> None:
    common = {
        "seed": 2027,
        "target_counts": (2, 3, 4),
        "split_fingerprint": "bootstrap",
        "dino_fingerprint": "dino",
        "catalog_fingerprint": "catalog",
        "training_fingerprint": "training",
    }
    identity = trainer._resume_identity(
        **common, formal_mode=False, debug_mode=True
    )
    assert identity["formal_mode"] is False
    assert identity["debug_mode"] is True

    with pytest.raises(ValueError, match="exactly one"):
        trainer._resume_identity(
            **common, formal_mode=False, debug_mode=False
        )


class _ToySelector(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = torch.nn.Linear(3, 2)


def _toy_training_state() -> tuple[
    _ToySelector,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
    torch.amp.GradScaler,
]:
    model = _ToySelector()
    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, optimizer, scheduler, scaler


def test_resume_round_trip_restores_all_rng_and_loader_generator(
    tmp_path: Path,
) -> None:
    trainer._set_seed(8317, deterministic=False)
    loader_generator = torch.Generator().manual_seed(1927)
    model, optimizer, scheduler, scaler = _toy_training_state()
    loss = model.decoder(torch.rand(4, 3)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    identity = trainer._resume_identity(
        seed=2025,
        target_counts=(41, 202, 404),
        split_fingerprint="bootstrap",
        dino_fingerprint="dino",
        catalog_fingerprint="catalog",
        training_fingerprint="training",
        formal_mode=True,
        debug_mode=False,
    )
    payload = trainer._resume_payload(
        epoch=3,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        identity=identity,
        loader_generator=loader_generator,
    )
    path = tmp_path / "selector_resume.pth"
    torch.save(payload, path)

    expected_python = [random.random() for _ in range(3)]
    expected_numpy = np.random.random(3)
    expected_torch = torch.rand(3)
    expected_loader = torch.randperm(17, generator=loader_generator)
    expected_cuda = None
    if torch.cuda.is_available():
        expected_cuda = torch.rand(3, device="cuda").cpu()

    for _ in range(5):
        random.random()
        np.random.random()
        torch.rand(1)
        torch.randperm(5, generator=loader_generator)
        if torch.cuda.is_available():
            torch.rand(1, device="cuda")

    resumed_model, resumed_optimizer, resumed_scheduler, resumed_scaler = (
        _toy_training_state()
    )
    resumed_generator = torch.Generator().manual_seed(1)
    start_epoch = trainer._load_resume(
        path,
        model=resumed_model,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        scaler=resumed_scaler,
        identity=identity,
        loader_generator=resumed_generator,
    )

    assert start_epoch == 4
    assert [random.random() for _ in range(3)] == expected_python
    np.testing.assert_array_equal(np.random.random(3), expected_numpy)
    torch.testing.assert_close(torch.rand(3), expected_torch, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.randperm(17, generator=resumed_generator),
        expected_loader,
        rtol=0,
        atol=0,
    )
    if expected_cuda is not None:
        torch.testing.assert_close(
            torch.rand(3, device="cuda").cpu(), expected_cuda, rtol=0, atol=0
        )


def test_resume_rejects_missing_or_corrupt_rng_state(tmp_path: Path) -> None:
    trainer._set_seed(9, deterministic=False)
    generator = torch.Generator().manual_seed(10)
    model, optimizer, scheduler, scaler = _toy_training_state()
    identity = trainer._resume_identity(
        seed=2026,
        target_counts=(41, 202, 404),
        split_fingerprint="bootstrap",
        dino_fingerprint="dino",
        catalog_fingerprint="catalog",
        training_fingerprint="training",
        formal_mode=True,
        debug_mode=False,
    )
    payload = trainer._resume_payload(
        epoch=1,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        identity=identity,
        loader_generator=generator,
    )

    for name, mutate, match in (
        (
            "missing",
            lambda value: value.pop("rng_state"),
            "rng_state mapping",
        ),
        (
            "loader",
            lambda value: value["rng_state"].__setitem__(
                "dataloader_generator", torch.tensor([1], dtype=torch.int64)
            ),
            "DataLoader generator state",
        ),
    ):
        candidate = tmp_path / f"{name}.pth"
        cloned = copy.deepcopy(payload)
        mutate(cloned)
        torch.save(cloned, candidate)
        resumed = _toy_training_state()
        with pytest.raises(ValueError, match=match):
            trainer._load_resume(
                candidate,
                model=resumed[0],
                optimizer=resumed[1],
                scheduler=resumed[2],
                scaler=resumed[3],
                identity=identity,
                loader_generator=torch.Generator(),
            )


def test_selector_output_namespace_rejects_v1_even_for_overwrite(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "selector"
    output_dir.mkdir()
    (output_dir / "selector_config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "pc_bpus_selector",
                "method": "PC-BPUS",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="namespace mismatch for schema_version"):
        trainer._assert_v2_selector_namespace(output_dir)

    (output_dir / "selector_config.json").write_text(
        json.dumps(
            {
                "schema_version": trainer.SELECTOR_SCHEMA_VERSION,
                "kind": trainer.SELECTOR_KIND,
                "method": trainer.SELECTOR_METHOD,
            }
        ),
        encoding="utf-8",
    )
    trainer._assert_v2_selector_namespace(output_dir)
