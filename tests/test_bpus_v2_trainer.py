from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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
        device="cuda",
        dry_run=False,
    )
    trainer._validate_formal_training_settings(args, formal_mode=True)
    args.batch_size = 4
    with pytest.raises(ValueError, match="--batch-size=8"):
        trainer._validate_formal_training_settings(args, formal_mode=True)
    trainer._validate_formal_training_settings(args, formal_mode=False)


def test_resume_identity_rejects_v1_payload() -> None:
    expected = trainer._resume_identity(
        seed=2025,
        target_counts=(41, 202, 404),
        split_fingerprint="bootstrap",
        dino_fingerprint="dino",
        catalog_fingerprint="catalog",
        training_fingerprint="training",
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
    )
    trainer._validate_resume_identity(expected, expected)
    for field in (
        "seed",
        "target_counts",
        "bootstrap_fingerprint",
        "dino_weight_fingerprint",
        "catalog_fingerprint",
        "training_fingerprint",
    ):
        tampered = dict(expected)
        tampered[field] = "tampered"
        with pytest.raises(ValueError, match=field):
            trainer._validate_resume_identity(tampered, expected)


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
