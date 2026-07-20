"""Labeled Base control trainer with every prototype component removed."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from Model.PC_HBM.training.losses import base_structure_loss
from utils.checkpoint_pc_hbm import (
    load_training_resume,
    save_decoder_checkpoint,
    save_training_resume,
)
from utils.dataloader import PCLabeledTrainDataset
from utils.distributed import is_main_process, reduce_mean, synchronize, unwrap_model


def configure_no_pc_base_trainability(model: nn.Module) -> tuple[str, ...]:
    """Freeze DINO/all PC branches and train only the original Decoder."""

    target = unwrap_model(model)
    target.requires_grad_(False)
    decoder = getattr(target, "decoder", None)
    if not isinstance(decoder, nn.Module):
        raise AttributeError("No-PC Base model must expose a Decoder module.")
    if getattr(decoder, "pc_hbm", None) is not None:
        raise RuntimeError("enabled=False requires decoder.pc_hbm is None.")
    if any(name.startswith("pc_hbm.") for name in decoder.state_dict()):
        raise RuntimeError("enabled=False Decoder must not contain pc_hbm.* state.")
    decoder.requires_grad_(True)
    dino = getattr(target, "dino", None)
    if isinstance(dino, nn.Module):
        dino.requires_grad_(False).eval()
    return tuple(
        f"decoder.{name}" for name, parameter in decoder.named_parameters()
        if parameter.requires_grad
    )


class NoPCBaseTrainer:
    """Train the frozen-DINO -> unchanged-Decoder Base ablation."""

    def __init__(
        self,
        model: nn.Module,
        cfg: Any,
        pc_cfg: Any,
        *,
        labeled_loader=None,
        optimizer=None,
        scheduler=None,
        scaler=None,
        decoder_warm_started: bool = False,
    ) -> None:
        if bool(getattr(pc_cfg, "enabled", True)):
            raise ValueError("NoPCBaseTrainer requires pc_cfg.enabled=False.")
        self.model = model
        self.model_without_ddp = unwrap_model(model)
        self.cfg = cfg
        self.pc_cfg = pc_cfg
        self.device = torch.device(getattr(cfg, "device", "cpu"))
        self.decoder = getattr(self.model_without_ddp, "decoder", None)
        configure_no_pc_base_trainability(self.model_without_ddp)
        if getattr(self.model_without_ddp, "encoder_pc_hbm", None) is not None:
            raise RuntimeError("enabled=False must not construct EncoderPCHBMAdapter.")
        if getattr(self.model_without_ddp, "pseudo_refiner", None) is not None:
            raise RuntimeError("enabled=False must not construct TeacherPseudoLabelRefiner.")

        learning_rate = 3.0e-5 if decoder_warm_started else float(
            getattr(cfg, "learning_rate", 1.0e-4)
        )
        self.optimizer = optimizer or torch.optim.Adam(
            [parameter for parameter in self.decoder.parameters() if parameter.requires_grad],
            lr=learning_rate,
            weight_decay=float(getattr(cfg, "weight_decay", 0.0)),
        )
        self.scheduler = scheduler or CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, int(getattr(cfg, "epochs", 30)) - 1),
            eta_min=float(getattr(cfg, "min_lr", 1.0e-7)),
        )
        self.amp_enabled = bool(
            getattr(cfg, "use_amp", True) and self.device.type == "cuda"
        )
        self.scaler = scaler or torch.amp.GradScaler(
            "cuda", enabled=self.amp_enabled
        )
        self.labeled_sampler = None
        self.labeled_loader = (
            labeled_loader if labeled_loader is not None else self._build_labeled_loader()
        )
        if len(self.labeled_loader) == 0:
            raise ValueError("No-PC Base labeled training loader is empty.")
        self.current_epoch = 1
        self.last_epoch_metrics: dict[str, float] = {}
        self.save_dir = Path(getattr(cfg, "save_dir", "./results/base_no_pc"))
        if is_main_process():
            self.save_dir.mkdir(parents=True, exist_ok=True)
        synchronize()

    def _build_labeled_loader(self):
        dataset = PCLabeledTrainDataset(
            l_image_root=self.cfg.train_imgs,
            l_gt_root=self.cfg.train_masks,
            l_txt_root=self.cfg.train_sample_txt,
            l_train_size=int(getattr(self.pc_cfg, "input_size", 392)),
            labeled_indices_pt=getattr(self.cfg, "train_labeled_indices_pt", None),
            rVFlip=bool(getattr(self.cfg, "rVFlip", True)),
            rCrop=bool(getattr(self.cfg, "rCrop", False)),
            rRotate=bool(getattr(self.cfg, "rRotate", False)),
            colorEnhance=bool(getattr(self.cfg, "colorEnhance", True)),
            rPeper=bool(getattr(self.cfg, "rPeper", False)),
        )
        sampler = None
        if bool(getattr(self.cfg, "distributed", False)):
            sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)
        self.labeled_sampler = sampler
        return DataLoader(
            dataset,
            batch_size=int(getattr(self.cfg, "batch_size", 16)),
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=True,
            num_workers=int(getattr(self.cfg, "num_workers", 0)),
            pin_memory=bool(getattr(self.cfg, "CUDA", self.device.type == "cuda")),
        )

    def train_epoch(self, epoch: int | None = None) -> dict[str, float]:
        epoch = self.current_epoch if epoch is None else int(epoch)
        if epoch != self.current_epoch:
            raise ValueError(
                f"train_epoch expected epoch {self.current_epoch}, received {epoch}."
            )
        self.model.train()
        configure_no_pc_base_trainability(self.model_without_ddp)
        getattr(self.model_without_ddp, "dino", nn.Identity()).eval()
        if self.labeled_sampler is not None:
            self.labeled_sampler.set_epoch(epoch)

        running: dict[str, float] = defaultdict(float)
        batch_count = 0
        for batch in self.labeled_loader:
            images, gt, _ = self._unpack_labeled_batch(batch)
            images = images.to(self.device, non_blocking=self.device.type == "cuda")
            gt = gt.to(self.device, non_blocking=self.device.type == "cuda")
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                outputs = self.model(
                    x=images,
                    memory=None,
                    pc_mode="off",
                    epoch=epoch,
                    return_aux=False,
                    query_image_ids=None,
                )
                loss = base_structure_loss(outputs, gt)
            if not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError(f"Non-finite no-PC Base loss at epoch {epoch}.")
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in self.decoder.parameters()
                    if parameter.requires_grad
                ],
                float(getattr(self.cfg, "grad_clip_norm", 5.0)),
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            running["loss"] += float(loss.detach())
            running["grad_norm"] += float(torch.as_tensor(grad_norm).detach())
            batch_count += 1

        if batch_count == 0:
            raise RuntimeError("No-PC Base labeled loader produced no batches.")
        metrics = {
            name: reduce_mean(value / batch_count, self.device)
            for name, value in running.items()
        }
        metrics["pc_enabled"] = 0.0
        self.last_epoch_metrics = metrics
        self.scheduler.step()
        self.current_epoch = epoch + 1
        interval = int(getattr(self.cfg, "checkpoint_interval", 1))
        if interval > 0 and epoch % interval == 0 and is_main_process():
            self._save_resume(epoch)
        synchronize()
        return metrics

    def train(self) -> None:
        final_epoch = int(getattr(self.cfg, "epochs", 30))
        while self.current_epoch <= final_epoch:
            self.train_epoch()
        if is_main_process():
            self._save_final(final_epoch)
        synchronize()

    def _save_resume(self, epoch: int) -> Path:
        path = self.save_dir / f"no_pc_base_resume_epoch_{epoch:03d}.pth"
        save_training_resume(
            path,
            epoch=epoch,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            pc_cfg=self.pc_cfg,
            extra={"model_role": "base_no_pc", "pc_enabled": False},
        )
        return path

    def _save_final(self, epoch: int) -> Path:
        path = self.save_dir / "no_pc_base_decoder.pth"
        save_decoder_checkpoint(
            path,
            self.decoder,
            self.pc_cfg,
            epoch,
            extra={"model_role": "base_no_pc", "pc_enabled": False},
        )
        return path

    def resume(self, path: str | Path, *, restore_rng: bool = True) -> dict[str, Any]:
        checkpoint = load_training_resume(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            restore_rng=restore_rng,
        )
        saved_cfg = checkpoint.get("pc_cfg", {})
        if not isinstance(saved_cfg, Mapping) or saved_cfg.get("enabled") is not False:
            raise RuntimeError("Resume checkpoint is not an enabled=False Base run.")
        self.current_epoch = int(checkpoint["epoch"]) + 1
        return checkpoint

    @staticmethod
    def _unpack_labeled_batch(
        batch: Sequence[Any] | Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        if isinstance(batch, Mapping):
            images = batch.get("image", batch.get("images"))
            gt = batch.get("gt", batch.get("mask"))
            image_ids = batch.get("image_id", batch.get("image_ids"))
        else:
            if len(batch) < 2:
                raise ValueError("Labeled batch must contain image and mask tensors.")
            images, gt = batch[0], batch[1]
            image_ids = batch[2] if len(batch) > 2 else None
        if not torch.is_tensor(images) or not torch.is_tensor(gt):
            raise TypeError("Labeled image and mask entries must be tensors.")
        if image_ids is None:
            ids = [str(index) for index in range(images.size(0))]
        elif isinstance(image_ids, str):
            ids = [image_ids]
        else:
            ids = [str(value) for value in image_ids]
        return images, gt, ids


__all__ = ["NoPCBaseTrainer", "configure_no_pc_base_trainability"]
