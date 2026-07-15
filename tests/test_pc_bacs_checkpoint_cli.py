from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from select_pc_bacs import _canonicalize_split_values, _load_selector_state
from utils.checkpoint_pc_hbm import (
    build_artifact_metadata,
    compute_labeled_split_fingerprint,
    save_decoder_checkpoint,
    state_dict_fingerprint,
)


class _TinyLegacyDecoder(nn.Module):
    """Minimal non-PC state for exercising the real checkpoint schema."""

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Conv2d(2, 1, kernel_size=1, bias=True)
        self.register_buffer("scale", torch.tensor(1.0))


@pytest.fixture
def selector_seed_keys() -> list[str]:
    return [f"TR-CAMO/seed_{index:02d}" for index in range(40)]


@pytest.fixture
def valid_selector_payload(tmp_path, selector_seed_keys):
    decoder = _TinyLegacyDecoder()
    metadata = build_artifact_metadata(
        training_design="two_stage",
        artifact_role="teacher_enhancer",
        labeled_split_fingerprint=compute_labeled_split_fingerprint(
            selector_seed_keys
        ),
        baseline_fingerprint="baseline-fingerprint",
        pc_frozen=True,
    )
    checkpoint_path = tmp_path / "teacher_enhancer.pth"
    payload = save_decoder_checkpoint(
        checkpoint_path,
        decoder,
        pc_cfg=None,
        epoch=5,
        artifact_meta=metadata,
    )
    return checkpoint_path, payload, decoder.state_dict()


def test_load_selector_state_accepts_strict_epoch5_two_stage_artifact(
    valid_selector_payload,
    selector_seed_keys,
) -> None:
    checkpoint_path, _, expected_state = valid_selector_payload

    legacy_state, metadata, non_pc_fingerprint, selector_fingerprint = (
        _load_selector_state(
            checkpoint_path,
            selector_seed_keys=selector_seed_keys,
            dino_fingerprint="dino-weight-fingerprint",
        )
    )

    assert set(legacy_state) == set(expected_state)
    for key, expected in expected_state.items():
        torch.testing.assert_close(legacy_state[key], expected)
    assert metadata["training_design"] == "two_stage"
    assert metadata["artifact_role"] == "teacher_enhancer"
    assert metadata["pc_frozen"] is True
    assert metadata["labeled_split_fingerprint"] == (
        compute_labeled_split_fingerprint(selector_seed_keys)
    )
    assert non_pc_fingerprint == state_dict_fingerprint(expected_state)
    assert len(selector_fingerprint) == 64


def test_load_selector_state_rejects_missing_decoder_weights(
    tmp_path,
    valid_selector_payload,
    selector_seed_keys,
) -> None:
    _, valid_payload, _ = valid_selector_payload
    payload = copy.deepcopy(valid_payload)
    payload["decoder"] = {}
    checkpoint_path = tmp_path / "missing_decoder_weights.pth"
    torch.save(payload, checkpoint_path)

    with pytest.raises((TypeError, RuntimeError), match="Decoder state_dict|no non-PC"):
        _load_selector_state(
            checkpoint_path,
            selector_seed_keys=selector_seed_keys,
            dino_fingerprint="dino-weight-fingerprint",
        )


def test_load_selector_state_rejects_wrong_epoch(
    tmp_path,
    valid_selector_payload,
    selector_seed_keys,
) -> None:
    _, valid_payload, _ = valid_selector_payload
    payload = copy.deepcopy(valid_payload)
    payload["epoch"] = 4
    checkpoint_path = tmp_path / "epoch4.pth"
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match="epoch-5"):
        _load_selector_state(
            checkpoint_path,
            selector_seed_keys=selector_seed_keys,
            dino_fingerprint="dino-weight-fingerprint",
        )


def test_load_selector_state_rejects_wrong_seed_fingerprint(
    valid_selector_payload,
    selector_seed_keys,
) -> None:
    checkpoint_path, _, _ = valid_selector_payload
    wrong_seed_keys = list(selector_seed_keys)
    wrong_seed_keys[-1] = "TR-COD10K/not_the_selector_seed"

    with pytest.raises(ValueError, match="labeled_split_fingerprint"):
        _load_selector_state(
            checkpoint_path,
            selector_seed_keys=wrong_seed_keys,
            dino_fingerprint="dino-weight-fingerprint",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("training_design", "joint"),
        ("artifact_role", "epoch_decoder"),
        ("pc_frozen", False),
    ],
)
def test_load_selector_state_rejects_wrong_artifact_identity(
    tmp_path,
    valid_selector_payload,
    selector_seed_keys,
    field,
    value,
) -> None:
    _, valid_payload, _ = valid_selector_payload
    payload = copy.deepcopy(valid_payload)
    payload["artifact_meta"][field] = value
    checkpoint_path = tmp_path / f"wrong_{field}.pth"
    torch.save(payload, checkpoint_path)

    with pytest.raises(ValueError, match=field):
        _load_selector_state(
            checkpoint_path,
            selector_seed_keys=selector_seed_keys,
            dino_fingerprint="dino-weight-fingerprint",
        )


def test_canonicalize_dataset_qualified_filename_with_duplicate_stem() -> None:
    catalog = ["TR-CAMO/foo", "TR-COD10K/foo", "TR-COD10K/bar"]

    assert _canonicalize_split_values(["TR-CAMO/foo.jpg"], catalog) == [
        "TR-CAMO/foo"
    ]


def test_canonicalize_unqualified_duplicate_stem_is_rejected() -> None:
    catalog = ["TR-CAMO/foo", "TR-COD10K/foo"]

    with pytest.raises(ValueError, match="uniquely resolve"):
        _canonicalize_split_values(["foo.jpg"], catalog)
