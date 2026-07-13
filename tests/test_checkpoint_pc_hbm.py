from pathlib import Path

import pytest
import torch
from torch import nn

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.PC_HBM.training.ema import update_ema_module
from utils.checkpoint_pc_hbm import (
    build_artifact_metadata,
    compute_labeled_split_fingerprint,
    compute_labeled_split_fingerprint_from_indices_pt,
    extract_non_pc_decoder_state,
    load_decoder_compatible,
    load_memory_checkpoint,
    load_training_resume,
    read_artifact_metadata,
    save_decoder_checkpoint,
    save_memory_checkpoint,
    save_training_resume,
    validate_artifact_metadata,
)


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(2, 2)
        self.pc_hbm = nn.Linear(2, 2)
        self.register_buffer("counter", torch.tensor(0))


class TinyMemory:
    def __init__(self):
        self.state = None

    def state_dict(self):
        return {
            "finalized": True,
            "compat_meta": {"architecture": "DINO_SCOD_PC_HBM", "schema_version": 1},
            "parent": {"p3_keys": torch.ones(1, 2, dtype=torch.float16)},
        }

    def load_state_dict(self, state):
        self.state = state.get("memory", state)

    def validate_compat(self, expected, require_producer_match=False):
        actual = self.state.get("compat_meta", {})
        for key, value in expected.items():
            if key == "producer_fingerprint" and not require_producer_match:
                continue
            if actual.get(key) != value:
                return False, f"{key} mismatch"
        return True, "ok"

    def is_ready(self):
        return bool(self.state and self.state.get("finalized"))


def test_legacy_decoder_allows_only_missing_pc_keys_and_module_prefix():
    source = TinyDecoder()
    legacy = {
        f"module.{key}": value.clone()
        for key, value in source.state_dict().items()
        if not key.startswith("pc_hbm.")
    }
    target = TinyDecoder()
    result = load_decoder_compatible(target, legacy)
    assert result.missing_keys and all(key.startswith("pc_hbm.") for key in result.missing_keys)

    partial = dict(legacy)
    partial["module.pc_hbm.weight"] = source.pc_hbm.weight.detach().clone()
    try:
        load_decoder_compatible(TinyDecoder(), partial)
    except RuntimeError as error:
        assert "Incomplete PC-HBM" in str(error)
    else:
        raise AssertionError("partial PC-HBM state must not be silently accepted")


def test_decoder_memory_and_resume_round_trip(tmp_path: Path):
    cfg = DinoPCHBMConfig()
    decoder = TinyDecoder()
    decoder_path = tmp_path / "decoder.pth"
    save_decoder_checkpoint(decoder_path, decoder, cfg, epoch=7)
    restored = TinyDecoder()
    load_decoder_compatible(restored, decoder_path, require_pc_complete=True)
    for expected, actual in zip(decoder.parameters(), restored.parameters()):
        torch.testing.assert_close(expected, actual)

    memory = TinyMemory()
    memory_path = tmp_path / "memory.pth"
    save_memory_checkpoint(memory_path, memory)
    loaded_memory = TinyMemory()
    load_memory_checkpoint(
        memory_path,
        loaded_memory,
        expected_compat={"architecture": "DINO_SCOD_PC_HBM", "schema_version": 1},
    )
    assert loaded_memory.is_ready()

    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    ema = TinyDecoder()
    resume_path = tmp_path / "resume.pth"
    save_training_resume(
        resume_path,
        epoch=3,
        model=decoder,
        optimizer=optimizer,
        ema_model=ema,
        pc_cfg=cfg,
    )
    decoder2, ema2 = TinyDecoder(), TinyDecoder()
    optimizer2 = torch.optim.Adam(decoder2.parameters(), lr=1e-3)
    checkpoint = load_training_resume(
        resume_path,
        model=decoder2,
        optimizer=optimizer2,
        ema_model=ema2,
    )
    assert checkpoint["epoch"] == 3


def test_named_ema_updates_parameters_and_copies_buffers():
    student, teacher = TinyDecoder(), TinyDecoder()
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.fill_(2.0)
        for parameter in teacher.parameters():
            parameter.zero_()
        student.counter.fill_(9)
    update_ema_module(student, teacher, momentum=0.5)
    assert all(torch.allclose(parameter, torch.ones_like(parameter)) for parameter in teacher.parameters())
    assert teacher.counter.item() == 9
    assert not teacher.training and not any(p.requires_grad for p in teacher.parameters())


def test_non_pc_state_filter_normalizes_prefix_and_keeps_buffers():
    decoder = TinyDecoder()
    nested = {
        "decoder": {f"module.{key}": value for key, value in decoder.state_dict().items()}
    }
    state = extract_non_pc_decoder_state(nested, clone=True)
    assert set(state) == {"base.weight", "base.bias", "counter"}
    assert not any(key.startswith("pc_hbm.") for key in state)
    assert state["counter"].data_ptr() != decoder.counter.data_ptr()


def test_labeled_split_fingerprint_is_key_normalized_and_order_independent(tmp_path: Path):
    expected = compute_labeled_split_fingerprint(["CAMO/a", "COD10K/b"])
    actual = compute_labeled_split_fingerprint(["./COD10K\\b", "CAMO/a", "CAMO/a"])
    assert actual == expected

    string_pt = tmp_path / "string_split.pt"
    torch.save(["COD10K\\b", "CAMO/a"], string_pt)
    assert compute_labeled_split_fingerprint_from_indices_pt(string_pt) == expected

    integer_pt = tmp_path / "integer_split.pt"
    torch.save(torch.tensor([1, 0, 1]), integer_pt)
    assert (
        compute_labeled_split_fingerprint_from_indices_pt(
            integer_pt, all_sample_keys=["CAMO/a", "COD10K/b"]
        )
        == expected
    )


def test_artifact_metadata_round_trip_and_design_validation(tmp_path: Path):
    split_fingerprint = compute_labeled_split_fingerprint(["CAMO/a"])
    metadata = build_artifact_metadata(
        training_design="teacher_only",
        artifact_role="teacher_enhancer",
        labeled_split_fingerprint=split_fingerprint,
        baseline_fingerprint="baseline-sha256",
        pc_frozen=False,
    )
    path = tmp_path / "teacher_enhancer.pth"
    save_decoder_checkpoint(
        path,
        TinyDecoder(),
        DinoPCHBMConfig(),
        epoch=30,
        artifact_meta=metadata,
    )
    assert read_artifact_metadata(path) == metadata
    validated = validate_artifact_metadata(
        path,
        {
            "training_design": "teacher_only",
            "artifact_role": "teacher_enhancer",
            "labeled_split_fingerprint": split_fingerprint,
        },
    )
    assert validated["baseline_fingerprint"] == "baseline-sha256"
    load_decoder_compatible(
        TinyDecoder(),
        path,
        require_pc_complete=True,
        expected_artifact_meta={"training_design": "teacher_only"},
    )

    with pytest.raises(RuntimeError, match="training_design"):
        validate_artifact_metadata(path, {"training_design": "joint"})


def test_two_stage_teacher_enhancer_is_a_valid_artifact_design(tmp_path: Path):
    split_fingerprint = compute_labeled_split_fingerprint(["CAMO/a"])
    path = tmp_path / "two_stage_teacher_enhancer.pth"
    metadata = build_artifact_metadata(
        training_design="two_stage",
        artifact_role="teacher_enhancer",
        labeled_split_fingerprint=split_fingerprint,
        baseline_fingerprint="stage-one-decoder-sha256",
        pc_frozen=True,
    )
    save_decoder_checkpoint(
        path,
        TinyDecoder(),
        DinoPCHBMConfig(),
        epoch=30,
        artifact_meta=metadata,
    )

    validated = validate_artifact_metadata(
        path,
        {
            "training_design": ("teacher_only", "two_stage"),
            "artifact_role": "teacher_enhancer",
            "labeled_split_fingerprint": split_fingerprint,
            "pc_frozen": True,
        },
    )
    assert validated == metadata


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    (
        ("artifact_role", "student_raw"),
        ("labeled_split_fingerprint", "different-split"),
        ("pc_frozen", False),
    ),
)
def test_two_stage_teacher_enhancer_keeps_strict_identity_validation(
    field, wrong_value
):
    metadata = build_artifact_metadata(
        training_design="two_stage",
        artifact_role="teacher_enhancer",
        labeled_split_fingerprint="split-sha256",
        baseline_fingerprint="stage-one-decoder-sha256",
        pc_frozen=True,
    )
    expected = {
        "training_design": ("teacher_only", "two_stage"),
        "artifact_role": "teacher_enhancer",
        "labeled_split_fingerprint": "split-sha256",
        "pc_frozen": True,
    }
    metadata[field] = wrong_value
    with pytest.raises(RuntimeError, match=field):
        validate_artifact_metadata({"artifact_meta": metadata}, expected)


def test_untagged_checkpoint_is_accepted_only_for_joint_design():
    legacy = {"decoder": TinyDecoder().state_dict()}
    assert validate_artifact_metadata(legacy, {"training_design": "joint"}) == {}
    with pytest.raises(RuntimeError, match="only with training_design='joint'"):
        validate_artifact_metadata(legacy, {"training_design": "teacher_only"})
    with pytest.raises(RuntimeError, match="only with training_design='joint'"):
        validate_artifact_metadata(
            legacy, {"training_design": ("teacher_only", "joint")}
        )


def test_incomplete_artifact_metadata_is_rejected():
    malformed = {
        "artifact_meta": {
            "training_design": "teacher_only",
            "artifact_role": "teacher_enhancer",
        }
    }
    with pytest.raises(RuntimeError, match="metadata is incomplete"):
        read_artifact_metadata(malformed)
