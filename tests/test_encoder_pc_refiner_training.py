from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from configs.pc_hbm_dino_config import EncoderPCHBMConfig
from Model.PC_HBM.encoder import TeacherPseudoLabelRefiner
import utils.trainer_base_model_encoder_pc as trainer_module


class _ReadyMemory:
    def is_ready(self) -> bool:
        return True


class _TinyDecoder(nn.Module):
    decoder_arch = "bgfbr_pc_v1"
    pc_hbm = None

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))


class _Stage21Model(nn.Module):
    def __init__(self, config: EncoderPCHBMConfig) -> None:
        super().__init__()
        self.encoder_pc_config = config
        self.encoder_pc_hbm = nn.Linear(1, 1, bias=False)
        self.decoder = _TinyDecoder()
        self.pseudo_refiner = TeacherPseudoLabelRefiner(config)
        self.dino = nn.Linear(1, 1)
        self.refiner_flags: list[bool] = []

    def forward(self, x, *, run_labeled_refiner=False, epoch=None, **kwargs):
        self.refiner_flags.append(bool(run_labeled_refiner))
        batch = x.size(0)
        adapter_value = self.encoder_pc_hbm.weight.reshape(1, 1, 1, 1)
        decoder_value = self.decoder.scale.reshape(1, 1, 1, 1)
        z_core = (adapter_value + decoder_value).expand(batch, 1, 98, 98)
        outputs = (z_core, z_core, z_core, z_core, z_core)
        aux: dict[str, object] = {
            "features": {
                "p1": decoder_value.expand(batch, 128, 98, 98),
            },
            "encoder_pc_hbm": {
                "refiner_evidence": {
                    "verified_evidence": adapter_value.expand(
                        batch, 128, 28, 28
                    ),
                    "boundary_probability": torch.full(
                        (batch, 1, 28, 28), 0.5, device=x.device
                    ),
                    "pc_gate": torch.ones(batch, 1, 28, 28, device=x.device),
                    "contradiction": torch.zeros(
                        batch, 1, 28, 28, device=x.device
                    ),
                    "semantic_support": torch.ones(
                        batch, 1, 28, 28, device=x.device
                    ),
                    "detail_support": torch.ones(
                        batch, 1, 28, 28, device=x.device
                    ),
                    "valid_map": torch.ones(
                        batch, 1, 28, 28, device=x.device
                    ),
                    "route_confidence": torch.full(
                        (batch,), 0.8, device=x.device
                    ),
                }
            },
        }
        if run_labeled_refiner:
            aux["pseudo_refiner"] = self.pseudo_refiner(
                z_core,
                aux["features"]["p1"],
                aux["encoder_pc_hbm"],
                epoch=epoch,
            )
        return outputs, aux


def test_base_epoch21_updates_only_refiner_from_detached_refiner_loss(
    monkeypatch, tmp_path
) -> None:
    config = EncoderPCHBMConfig()
    model = _Stage21Model(config)
    trainable_modules = (
        model.encoder_pc_hbm,
        model.decoder,
        model.pseudo_refiner,
    )
    optimizer = torch.optim.Adam(
        [
            parameter
            for module in trainable_modules
            for parameter in module.parameters()
        ],
        lr=1.0e-3,
    )

    def configure(adapter, decoder, pseudo_refiner, stage):
        adapter.requires_grad_(True)
        decoder.requires_grad_(True)
        pseudo_refiner.requires_grad_(stage.enable_refiner)
        return {}

    def zero_core_loss(outputs, aux, gt, pc_cfg, stage):
        loss = outputs[3].sum() * 0.0
        return loss, {"L_core_zero": loss.detach()}

    monkeypatch.setattr(trainer_module, "configure_encoder_pc_stage", configure)
    monkeypatch.setattr(trainer_module, "encoder_pc_labeled_loss", zero_core_loss)
    monkeypatch.setattr(
        trainer_module, "update_ema_encoder_adapter", lambda *args, **kwargs: None
    )
    cfg = SimpleNamespace(
        device="cpu",
        use_amp=False,
        epochs=30,
        min_lr=1.0e-6,
        grad_clip_norm=5.0,
        checkpoint_interval=0,
        save_dir=str(tmp_path),
    )
    loader = [
        (
            torch.randn(1, 3, 8, 8),
            torch.randint(0, 2, (1, 1, 98, 98)).float(),
            ["labeled-0"],
        )
    ]
    trainer = trainer_module.EncoderPCHBMTrainer(
        model,
        cfg,
        config,
        memory=_ReadyMemory(),
        labeled_loader=loader,
        memory_loader=loader,
        optimizer=optimizer,
        ema_adapter=nn.Linear(1, 1, bias=False),
        memory_rebuild_fn=lambda *args, **kwargs: None,
    )
    trainer.current_epoch = 21

    metrics = trainer.train_epoch(21)

    assert model.refiner_flags == [True]
    assert "L_refiner_total" in metrics
    assert all(parameter.grad is None for parameter in model.dino.parameters())
    for module in (model.encoder_pc_hbm, model.decoder):
        assert all(
            parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
            for parameter in module.parameters()
        )
    refiner_gradients = [
        parameter.grad
        for parameter in model.pseudo_refiner.parameters()
        if parameter.grad is not None
    ]
    assert refiner_gradients
    assert any(torch.count_nonzero(gradient) > 0 for gradient in refiner_gradients)
