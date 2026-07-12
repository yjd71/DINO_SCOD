"""Dedicated staged Base trainer for DINO PC-HBM.

The legacy :mod:`utils.trainer_base_model` trainer intentionally remains
unchanged.  This module owns the PC-HBM-only lifecycle: decoder optimization,
the frozen EMA memory producer, labeled-memory rebuilds, strict staged losses,
diagnostics, AMP and resumable checkpoints.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from Model.PC_HBM.memory import PCMemory
from Model.PC_HBM.training import (
    DiagnosticWarningTracker,
    collect_pc_diagnostics,
    make_ema_copy,
    pc_hbm_labeled_loss,
    pc_mode_for_epoch,
    update_ema_module,
)
from configs.pc_hbm_dino_config import DinoPCHBMConfig
from utils.checkpoint_pc_hbm import (
    load_training_resume,
    save_decoder_checkpoint,
    save_memory_checkpoint,
    save_training_resume,
)
from utils.dataloader import PCLabeledTrainDataset
from utils.distributed import is_main_process, reduce_mean, synchronize, unwrap_model
from utils.pc_memory_runner import build_labeled_memory_loader, rebuild_memory


def current_time() -> str:
    return datetime.now().strftime("%m-%d %H:%M:%S")


class BasePCHBMTrainer:
    """Train only the Decoder with the locked 1-based PC-HBM schedule."""

    def __init__(
        self,
        model: nn.Module,
        cfg: Any,
        pc_cfg: DinoPCHBMConfig | None = None,
        *,
        memory: PCMemory | Any | None = None,
        labeled_loader=None,
        memory_loader=None,
        optimizer=None,
        scheduler=None,
        scaler=None,
        memory_decoder: nn.Module | None = None,
        memory_rebuild_fn: Callable[..., Any] = rebuild_memory,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.pc_cfg = pc_cfg or getattr(cfg, "pc_hbm", None) or DinoPCHBMConfig()
        self.device = torch.device(getattr(cfg, "device", "cpu"))
        self.distributed = bool(getattr(cfg, "distributed", False))
        self.model_without_ddp = unwrap_model(model)
        if not hasattr(self.model_without_ddp, "decoder"):
            raise AttributeError("Base PC-HBM model must expose a Decoder as model.decoder")
        self.decoder = self.model_without_ddp.decoder
        if getattr(self.decoder, "pc_hbm", None) is None:
            raise RuntimeError("Base PC-HBM trainer requires BaseModel(pc_cfg=DinoPCHBMConfig(...))")

        self.optimizer = optimizer or Adam(
            (parameter for parameter in self.decoder.parameters() if parameter.requires_grad),
            lr=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
        )
        self._validate_decoder_only_optimizer()
        self.scheduler = scheduler or CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, int(cfg.epochs) - 1),
            eta_min=float(cfg.min_lr),
        )

        self.amp_enabled = bool(
            getattr(self.pc_cfg, "use_amp", True) and self.device.type == "cuda"
        )
        self.scaler = scaler or torch.amp.GradScaler("cuda", enabled=self.amp_enabled)
        self.memory_decoder = memory_decoder or make_ema_copy(self.decoder)
        self.memory_decoder.to(self.device).eval().requires_grad_(False)
        self._validate_ema_schema()

        self.memory = memory or PCMemory(config=self.pc_cfg)
        self.memory_rebuild_fn = memory_rebuild_fn
        self.labeled_train_set = None
        self.labeled_sampler = None
        self.labeled_train_dl = labeled_loader or self._build_labeled_loader()
        self.memory_loader = memory_loader or self._build_memory_loader()
        if len(self.labeled_train_dl) == 0:
            raise ValueError("PC-HBM labeled training loader is empty")

        self.warning_tracker = DiagnosticWarningTracker(self.pc_cfg)
        self.current_epoch = 1
        self.last_epoch_metrics: dict[str, float] = {}
        self.save_dir = Path(cfg.save_dir)
        if is_main_process():
            self.save_dir.mkdir(parents=True, exist_ok=True)
        synchronize()

    def _build_labeled_loader(self):
        self.labeled_train_set = PCLabeledTrainDataset(
            l_image_root=self.cfg.train_imgs,
            l_gt_root=self.cfg.train_masks,
            l_txt_root=self.cfg.train_sample_txt,
            l_train_size=self.cfg.train_size,
            labeled_indices_pt=self.cfg.train_labeled_indices_pt,
            rVFlip=bool(getattr(self.cfg, "rVFlip", True)),
            rCrop=bool(getattr(self.cfg, "rCrop", True)),
            rRotate=bool(getattr(self.cfg, "rRotate", False)),
            colorEnhance=bool(getattr(self.cfg, "colorEnhance", True)),
            rPeper=bool(getattr(self.cfg, "rPeper", False)),
        )
        if len(self.labeled_train_set) == 0:
            raise ValueError("PC-HBM labeled training set is empty")
        if self.distributed:
            self.labeled_sampler = DistributedSampler(
                self.labeled_train_set,
                num_replicas=int(self.cfg.world_size),
                rank=int(self.cfg.rank),
                shuffle=True,
                seed=int(self.cfg.seed),
            )
        return DataLoader(
            self.labeled_train_set,
            batch_size=int(self.cfg.batch_size),
            shuffle=self.labeled_sampler is None,
            sampler=self.labeled_sampler,
            num_workers=int(self.cfg.num_workers),
            pin_memory=bool(getattr(self.cfg, "CUDA", False)),
            persistent_workers=int(self.cfg.num_workers) > 0,
            drop_last=False,
        )

    def _build_memory_loader(self):
        return build_labeled_memory_loader(
            l_image_root=self.cfg.train_imgs,
            l_gt_root=self.cfg.train_masks,
            l_txt_root=self.cfg.train_sample_txt,
            l_train_size=self.cfg.train_size,
            labeled_indices_pt=self.cfg.train_labeled_indices_pt,
            batch_size=int(getattr(self.cfg, "memory_batch_size", self.cfg.batch_size)),
            num_workers=int(getattr(self.cfg, "memory_num_workers", self.cfg.num_workers)),
            pin_memory=bool(getattr(self.cfg, "CUDA", False)),
        )

    def _validate_decoder_only_optimizer(self) -> None:
        expected = {id(parameter) for parameter in self.decoder.parameters() if parameter.requires_grad}
        actual = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        }
        if actual != expected:
            missing = len(expected - actual)
            extra = len(actual - expected)
            raise RuntimeError(
                "Base PC-HBM optimizer must contain every trainable Decoder parameter "
                f"and nothing else (missing={missing}, extra={extra})"
            )

    def _validate_ema_schema(self) -> None:
        decoder_parameters = dict(self.decoder.named_parameters())
        ema_parameters = dict(self.memory_decoder.named_parameters())
        decoder_buffers = dict(self.decoder.named_buffers())
        ema_buffers = dict(self.memory_decoder.named_buffers())
        if decoder_parameters.keys() != ema_parameters.keys():
            raise RuntimeError("Decoder and memory_decoder parameter keys differ")
        if decoder_buffers.keys() != ema_buffers.keys():
            raise RuntimeError("Decoder and memory_decoder buffer keys differ")

    def _rebuild_epoch_memory(self, epoch: int) -> None:
        if int(epoch) < int(self.pc_cfg.parent_start_epoch):
            return
        self.memory_rebuild_fn(
            model=self.model,
            memory_decoder=self.memory_decoder,
            memory_loader=self.memory_loader,
            memory=self.memory,
            device=self.device,
            config=self.pc_cfg,
            use_amp=self.amp_enabled,
        )
        self._assert_memory_ready(epoch)
        synchronize()

    def _assert_memory_ready(self, epoch: int) -> None:
        if not hasattr(self.memory, "is_ready") or not bool(self.memory.is_ready()):
            raise RuntimeError(
                f"Epoch {epoch} mode={pc_mode_for_epoch(epoch, self.pc_cfg)!r} "
                "requires a finalized compatible labeled PC-HBM memory"
            )
        if hasattr(self.memory, "validate_compat"):
            compatibility = self.memory.validate_compat(self.pc_cfg.expected_memory_meta())
            if not bool(compatibility):
                reason = getattr(compatibility, "reason", "compatibility validation failed")
                raise RuntimeError(f"Epoch {epoch} PC-HBM memory is incompatible: {reason}")

    def train_epoch(self, epoch: int | None = None) -> dict[str, float]:
        epoch = self.current_epoch if epoch is None else int(epoch)
        if epoch != self.current_epoch:
            raise ValueError(
                f"train_epoch expected current epoch {self.current_epoch}, received {epoch}"
            )
        mode = pc_mode_for_epoch(epoch, self.pc_cfg)
        self._rebuild_epoch_memory(epoch)
        if mode != "off":
            self._assert_memory_ready(epoch)

        self.model.train()
        self.memory_decoder.eval()
        running: dict[str, float] = defaultdict(float)
        batch_count = 0
        iterator = tqdm(self.labeled_train_dl, disable=not is_main_process())
        for iteration, batch in enumerate(iterator, start=1):
            images, gt, image_ids = self._unpack_labeled_batch(batch)
            images = images.to(self.device, non_blocking=bool(getattr(self.cfg, "CUDA", False)))
            gt = gt.to(self.device, non_blocking=bool(getattr(self.cfg, "CUDA", False)))
            self.optimizer.zero_grad(set_to_none=True)

            active_memory = None if mode == "off" else self.memory
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                result = self.model(
                    x=images,
                    memory=active_memory,
                    pc_mode=mode,
                    epoch=epoch,
                    return_aux=True,
                    query_image_ids=image_ids,
                )
                if not isinstance(result, (tuple, list)) or len(result) != 2:
                    raise RuntimeError("PC-HBM model must return (outputs, aux) with return_aux=True")
                outputs, aux = result
                loss, loss_metrics = pc_hbm_labeled_loss(
                    outputs,
                    aux,
                    gt,
                    epoch,
                    self.pc_cfg,
                    pc_mode=mode,
                    strict=True,
                )
            if not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError(
                    f"Non-finite Base PC-HBM loss at epoch={epoch}, iteration={iteration}"
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.decoder.parameters(), float(self.pc_cfg.grad_clip_norm)
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            update_ema_module(
                self.decoder,
                self.memory_decoder,
                momentum=float(self.pc_cfg.ema_momentum),
            )

            diagnostics = collect_pc_diagnostics(aux, gt)
            batch_metrics = {
                **loss_metrics,
                **diagnostics,
                "loss": loss.detach(),
                "grad_norm": torch.as_tensor(grad_norm).detach(),
            }
            self._accumulate_metrics(running, batch_metrics)
            batch_count += 1
            if is_main_process() and iteration % int(self.cfg.log_interval) == 0:
                printable = {
                    key: self._scalar(value)
                    for key, value in batch_metrics.items()
                    if self._is_scalar(value)
                }
                print(
                    f"[Base PC-HBM] epoch={epoch} mode={mode} iteration={iteration} "
                    + self._format_metrics(printable)
                )

            del outputs, aux, loss

        if batch_count == 0:
            raise RuntimeError("PC-HBM labeled training loader produced no batches")
        epoch_metrics = {
            name: reduce_mean(total / batch_count, self.device)
            for name, total in sorted(running.items())
        }
        self.last_epoch_metrics = epoch_metrics
        self.warning_tracker.update(epoch_metrics, emit=is_main_process())
        used_lr = float(self.optimizer.param_groups[0]["lr"])
        if self.scheduler is not None:
            self.scheduler.step()
        self.current_epoch = epoch + 1
        self._save_epoch(epoch, epoch_metrics)
        if is_main_process():
            print(
                f"[Base PC-HBM Epoch] epoch={epoch} mode={mode} lr={used_lr:.8g} "
                + self._format_metrics(epoch_metrics)
            )
        synchronize()
        return epoch_metrics

    def train(self) -> None:
        if is_main_process():
            sample_count = (
                len(self.labeled_train_set)
                if self.labeled_train_set is not None
                else "injected-loader"
            )
            print(f"<<< Start Base PC-HBM training; labeled={sample_count}")
        while self.current_epoch <= int(self.cfg.epochs):
            epoch = self.current_epoch
            if self.labeled_sampler is not None:
                self.labeled_sampler.set_epoch(epoch - 1)
            if is_main_process():
                print(f"{current_time()} >>> Epoch {epoch}/{self.cfg.epochs}")
            self.train_epoch(epoch)
        if is_main_process():
            print("<<< Base PC-HBM training finished")

    def resume(self, path: str | os.PathLike, *, restore_rng: bool = True) -> dict[str, Any]:
        checkpoint = load_training_resume(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            ema_model=self.memory_decoder,
            restore_rng=restore_rng,
        )
        self._validate_resume_config(checkpoint.get("pc_cfg"))
        completed_epoch = int(checkpoint.get("epoch", 0))
        if completed_epoch < 0 or completed_epoch > int(self.cfg.epochs):
            raise RuntimeError(f"Invalid resume epoch: {completed_epoch}")
        self.current_epoch = completed_epoch + 1
        history = (checkpoint.get("extra") or {}).get("diagnostic_history", {})
        if isinstance(history, Mapping):
            for name, values in history.items():
                if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                    self.warning_tracker.history[str(name)].extend(float(value) for value in values)
        return checkpoint

    def _save_epoch(self, epoch: int, epoch_metrics: Mapping[str, float]) -> None:
        if not is_main_process():
            return
        interval = max(1, int(getattr(self.cfg, "checkpoint_interval", 1)))
        should_export = epoch % interval == 0 or epoch == int(self.cfg.epochs)
        diagnostic_history = {
            name: list(values) for name, values in self.warning_tracker.history.items()
        }
        save_training_resume(
            self.save_dir / "training_resume.pth",
            epoch=epoch,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            ema_model=self.memory_decoder,
            pc_cfg=self.pc_cfg,
            extra={
                "epoch_metrics": dict(epoch_metrics),
                "diagnostic_history": diagnostic_history,
            },
        )
        if not should_export:
            return
        save_decoder_checkpoint(
            self.save_dir / f"base_pc_hbm_decoder_epoch_{epoch}.pth",
            self.decoder,
            self.pc_cfg,
            epoch,
        )
        if hasattr(self.memory, "is_ready") and bool(self.memory.is_ready()):
            save_memory_checkpoint(
                self.save_dir / f"base_pc_hbm_memory_epoch_{epoch}.pth",
                self.memory,
            )

    def _validate_resume_config(self, saved_config: Any) -> None:
        if saved_config is None:
            raise RuntimeError("PC-HBM resume checkpoint has no pc_cfg")
        current = asdict(self.pc_cfg) if is_dataclass(self.pc_cfg) else dict(vars(self.pc_cfg))
        saved = dict(saved_config)
        if saved != current:
            differing = sorted(
                key
                for key in set(saved) | set(current)
                if saved.get(key) != current.get(key)
            )
            raise RuntimeError(f"Resume PC-HBM config differs for keys: {differing}")

    @staticmethod
    def _unpack_labeled_batch(batch) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        if isinstance(batch, Mapping):
            images = batch.get("images", batch.get("image"))
            gt = batch.get("gt", batch.get("masks"))
            image_ids = batch.get("image_ids", batch.get("sample_keys"))
        elif isinstance(batch, (tuple, list)) and len(batch) == 4:
            _, images, gt, image_ids = batch
        else:
            raise TypeError(
                "Base PC-HBM labeled batches must contain images, GT and stable image ids"
            )
        if not torch.is_tensor(images) or not torch.is_tensor(gt):
            raise TypeError("Base PC-HBM labeled images and GT must be tensors")
        if isinstance(image_ids, str):
            image_ids = [image_ids]
        elif isinstance(image_ids, Sequence):
            image_ids = [str(value) for value in image_ids]
        else:
            raise TypeError("Base PC-HBM image ids must be strings or a string sequence")
        if len(image_ids) != images.size(0):
            raise ValueError("Base PC-HBM image id count differs from image batch size")
        return images, gt, image_ids

    @classmethod
    def _accumulate_metrics(cls, running: dict[str, float], metrics: Mapping[str, Any]) -> None:
        for name, value in metrics.items():
            if cls._is_scalar(value):
                running[name] += cls._scalar(value)

    @staticmethod
    def _is_scalar(value: Any) -> bool:
        return isinstance(value, (int, float)) or (
            torch.is_tensor(value) and value.numel() == 1
        )

    @staticmethod
    def _scalar(value: Any) -> float:
        if torch.is_tensor(value):
            return float(value.detach().float().cpu())
        return float(value)

    @staticmethod
    def _format_metrics(metrics: Mapping[str, float]) -> str:
        return " ".join(f"{name}={float(value):.5g}" for name, value in sorted(metrics.items()))


# Concise import-compatible alias for the dedicated entry point.
Trainer = BasePCHBMTrainer


__all__ = ["BasePCHBMTrainer", "Trainer", "current_time"]
