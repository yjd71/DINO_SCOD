from pathlib import Path

import torch
from torch import nn

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.PC_HBM.training.ema import update_ema_module
from utils.checkpoint_pc_hbm import (
    load_decoder_compatible,
    load_memory_checkpoint,
    load_training_resume,
    save_decoder_checkpoint,
    save_memory_checkpoint,
    save_training_resume,
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
