from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from configs.pc_hbm_dino_config import DinoPCHBMConfig
import utils.trainer_base_model_pc_hbm as trainer_module
from utils.trainer_base_model_pc_hbm import BasePCHBMTrainer


class _FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))
        self.register_buffer("seen", torch.tensor(0))
        self.pc_hbm = True


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = _FakeDecoder()
        self.backbone = nn.Parameter(torch.tensor(7.0))

    def forward(
        self,
        x,
        memory=None,
        pc_mode="off",
        epoch=None,
        return_aux=False,
        query_image_ids=None,
    ):
        del memory, epoch, query_image_ids
        z = self.decoder.weight * torch.ones(x.size(0), 1, 8, 8)
        outputs = (z, z, z, z, z)
        aux = {
            "z_main": z,
            "z_final": z,
            "p_final": torch.sigmoid(z),
            "pc_active": pc_mode != "off",
            "fallback_reason": None,
            "forward_mode": pc_mode,
            "pc_hbm": {"B3": z},
            "p2_bra": {},
            "p1_pra": {},
            "mixture": {},
        }
        return (outputs, aux) if return_aux else outputs


class _FakeMemory:
    def __init__(self):
        self.ready = False
        self.rebuilds = 0

    def is_ready(self):
        return self.ready

    def validate_compat(self, expected):
        del expected
        return True

    def state_dict(self):
        return {"compat_meta": {}, "ready": self.ready}


def _cfg(tmp_path):
    return SimpleNamespace(
        device=torch.device("cpu"),
        distributed=False,
        learning_rate=1.0e-2,
        weight_decay=0.0,
        min_lr=1.0e-5,
        epochs=6,
        batch_size=2,
        num_workers=0,
        CUDA=False,
        save_dir=str(tmp_path),
        log_interval=100,
        checkpoint_interval=1,
    )


def _diagnostic_trainer(pc_cfg=None):
    trainer = BasePCHBMTrainer.__new__(BasePCHBMTrainer)
    trainer.pc_cfg = pc_cfg or DinoPCHBMConfig(use_amp=False, diagnostic_window_epochs=3)
    trainer.warning_tracker = trainer_module.DiagnosticWarningTracker(trainer.pc_cfg)
    trainer._diagnostic_mode = None
    return trainer


def _collapsed_metrics():
    return {
        "pi_keep_mean": 1.0,
        "pi_res_mean": 0.0,
        "pi_def_mean": 0.0,
        "pi_sup_mean": 0.0,
        "gate_pc_mean": 0.0,
        "child_verify_auc": 0.5,
    }


def test_diagnostic_tracker_ignores_off_and_parent_only_epochs():
    trainer = _diagnostic_trainer()
    metrics = _collapsed_metrics()

    for mode in ("off", "parent_only"):
        trainer._set_diagnostic_mode(mode)
        assert trainer._update_warning_tracker(mode, metrics, emit=False) == []

    assert trainer._diagnostic_mode == "parent_only"
    assert dict(trainer.warning_tracker.history) == {}


def test_first_full_epoch_resets_history_and_requires_full_window():
    trainer = _diagnostic_trainer()
    metrics = _collapsed_metrics()
    trainer._diagnostic_mode = "parent_only"
    for name, value in metrics.items():
        trainer.warning_tracker.history[name].extend([value, value, value])

    trainer._set_diagnostic_mode("full")
    assert dict(trainer.warning_tracker.history) == {}
    assert trainer._update_warning_tracker("full", metrics, emit=False) == []
    assert trainer._update_warning_tracker("full", metrics, emit=False) == []
    messages = trainer._update_warning_tracker("full", metrics, emit=False)

    assert messages
    assert any("mixture collapse" in message for message in messages)


def test_resume_restores_only_full_mode_diagnostic_history(monkeypatch):
    checkpoint = {
        "epoch": 6,
        "pc_cfg": {},
        "extra": {"diagnostic_history": {"pi_keep_mean": [1.0, 1.0, 1.0]}},
    }

    def fake_load_training_resume(*args, **kwargs):
        del args, kwargs
        return checkpoint

    monkeypatch.setattr(trainer_module, "load_training_resume", fake_load_training_resume)

    def make_resume_trainer():
        trainer = _diagnostic_trainer()
        trainer.model = nn.Identity()
        trainer.optimizer = None
        trainer.scheduler = None
        trainer.scaler = None
        trainer.memory_decoder = nn.Identity()
        trainer.cfg = SimpleNamespace(epochs=30)
        trainer._validate_resume_config = lambda saved_config: None
        trainer.warning_tracker.history["gate_pc_mean"].extend([0.0, 0.0, 0.0])
        return trainer

    parent_only = make_resume_trainer()
    parent_only.resume("unused.pth", restore_rng=False)
    assert parent_only.current_epoch == 7
    assert parent_only._diagnostic_mode == "parent_only"
    assert dict(parent_only.warning_tracker.history) == {}

    checkpoint["epoch"] = 11
    full = make_resume_trainer()
    full.resume("unused.pth", restore_rng=False)
    assert full.current_epoch == 12
    assert full._diagnostic_mode == "full"
    assert list(full.warning_tracker.history["pi_keep_mean"]) == [1.0, 1.0, 1.0]
    assert "gate_pc_mean" not in full.warning_tracker.history


def test_parent_only_rebuild_decoder_only_ema_checkpoint_and_resume(tmp_path):
    cfg = _cfg(tmp_path)
    pc_cfg = DinoPCHBMConfig(use_amp=False)
    images = torch.randn(2, 3, 8, 8)
    gt = torch.randint(0, 2, (2, 1, 8, 8)).float()
    loader = [(images.clone(), images, gt, ["a", "b"])]
    memory = _FakeMemory()

    def fake_rebuild(**kwargs):
        kwargs["memory"].ready = True
        kwargs["memory"].rebuilds += 1
        return kwargs["memory"]

    model = _FakeModel()
    backbone_before = model.backbone.detach().clone()
    decoder_before = model.decoder.weight.detach().clone()
    trainer = BasePCHBMTrainer(
        model,
        cfg,
        pc_cfg,
        memory=memory,
        labeled_loader=loader,
        memory_loader=[None],
        memory_rebuild_fn=fake_rebuild,
    )
    trainer.current_epoch = 6
    metrics = trainer.train_epoch(6)

    assert memory.rebuilds == 1 and memory.ready
    assert torch.equal(model.backbone.detach(), backbone_before)
    assert not torch.equal(model.decoder.weight.detach(), decoder_before)
    assert trainer.current_epoch == 7
    assert torch.isfinite(torch.tensor(metrics["loss"]))
    for filename in (
        "training_resume.pth",
        "base_pc_hbm_decoder_epoch_6.pth",
        "base_pc_hbm_memory_epoch_6.pth",
    ):
        assert Path(tmp_path, filename).is_file()

    resumed = BasePCHBMTrainer(
        _FakeModel(),
        cfg,
        pc_cfg,
        memory=_FakeMemory(),
        labeled_loader=loader,
        memory_loader=[None],
        memory_rebuild_fn=fake_rebuild,
    )
    resumed.resume(Path(tmp_path, "training_resume.pth"), restore_rng=False)
    assert resumed.current_epoch == 7
