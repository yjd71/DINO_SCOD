"""Online pseudo-label trainer for the DINO PC-HBM teacher/student model.

This module is deliberately separate from the legacy pseudo and SAM trainers.
It keeps the two Student passes visible to DDP, rebuilds labeled-only memory
from the frozen EMA Teacher at every epoch boundary, and never writes pseudo
labels back into memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from Model.PC_HBM.memory import PCMemory
from Model.PC_HBM.training import (
    pc_hbm_labeled_loss,
    pc_unlabeled_loss,
    prepare_pseudo_targets,
    update_ema_module,
)
from utils.checkpoint_pc_hbm import (
    load_training_resume,
    save_decoder_checkpoint,
    save_memory_checkpoint,
    save_training_resume,
)
from utils.dataloader import (
    PCLabeledTrainDataset,
    UnlabeledPseudoTrainDataset,
    build_labeled_memory_loader,
)
from utils.distributed import (
    is_main_process,
    reduce_mean,
    synchronize,
    unwrap_model,
)
from utils.pc_memory_runner import build_memory_compat_meta, rebuild_memory


class PCHBMPseudoTrainer:
    """Train the PC-HBM Student with labeled and online-pseudo batches."""

    def __init__(
        self,
        model,
        cfg,
        pc_cfg,
        *,
        memory=None,
        scheduler=None,
        resume_path=None,
    ):
        self.model = model
        self.cfg = cfg
        self.pc_cfg = pc_cfg
        self.distributed = bool(getattr(cfg, "distributed", False))
        self.device = torch.device(cfg.device)
        self.core_model = unwrap_model(model)
        self._validate_model_contract()

        # The published TS protocol fixes both physical batches at 32.  DDP
        # partitions data across ranks but does not change the per-rank batch.
        if int(cfg.l_batch_size) != 32 or int(cfg.u_batch_size) != 32:
            raise ValueError("PC-HBM TS requires physical labeled/unlabeled batches of 32")

        student_parameters = [
            parameter
            for parameter in self.core_model.student.parameters()
            if parameter.requires_grad
        ]
        if not student_parameters:
            raise RuntimeError("PC-HBM Student has no trainable parameters")
        self.optimizer = optim.Adam(
            student_parameters,
            lr=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
        )
        self.scheduler = scheduler or optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, int(cfg.epochs)),
            eta_min=float(cfg.min_lr),
        )
        self.amp_enabled = bool(
            getattr(pc_cfg, "use_amp", True) and self.device.type == "cuda"
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        self.labeled_train_set = PCLabeledTrainDataset(
            l_image_root=cfg.train_imgs,
            l_gt_root=cfg.train_masks,
            l_txt_root=cfg.train_sample_txt,
            l_train_size=cfg.l_train_size,
            labeled_indices_pt=cfg.train_labeled_indices_pt,
            rVFlip=True,
            rCrop=True,
            rRotate=False,
            colorEnhance=True,
            rPeper=False,
        )
        self.unlabeled_train_set = UnlabeledPseudoTrainDataset(
            u_image_root=cfg.train_imgs,
            sampled_txt=cfg.train_sample_txt,
            u_train_size=cfg.u_train_size,
            labeled_indices_pt=cfg.train_labeled_indices_pt,
        )
        if len(self.labeled_train_set) < int(cfg.l_batch_size):
            raise ValueError("Labeled set is smaller than the fixed PC-HBM batch of 32")
        if len(self.unlabeled_train_set) < int(cfg.u_batch_size):
            raise ValueError("Unlabeled set is smaller than the fixed PC-HBM batch of 32")

        self.labeled_sampler = self._distributed_sampler(self.labeled_train_set)
        self.unlabeled_sampler = self._distributed_sampler(self.unlabeled_train_set)
        loader_kwargs = {
            "num_workers": int(cfg.num_workers),
            "pin_memory": bool(cfg.CUDA),
            "persistent_workers": int(cfg.num_workers) > 0,
            "drop_last": True,
        }
        self.labeled_train_dl = DataLoader(
            self.labeled_train_set,
            batch_size=32,
            shuffle=self.labeled_sampler is None,
            sampler=self.labeled_sampler,
            **loader_kwargs,
        )
        self.unlabeled_train_dl = DataLoader(
            self.unlabeled_train_set,
            batch_size=32,
            shuffle=self.unlabeled_sampler is None,
            sampler=self.unlabeled_sampler,
            **loader_kwargs,
        )
        if len(self.labeled_train_dl) == 0 or len(self.unlabeled_train_dl) == 0:
            raise RuntimeError("PC-HBM TS loaders must each contain at least one full batch")

        self.memory_loader = build_labeled_memory_loader(
            l_image_root=cfg.train_imgs,
            l_gt_root=cfg.train_masks,
            l_txt_root=cfg.train_sample_txt,
            l_train_size=cfg.l_train_size,
            labeled_indices_pt=cfg.train_labeled_indices_pt,
            batch_size=int(getattr(cfg, "memory_batch_size", 16)),
            num_workers=int(getattr(cfg, "memory_num_workers", cfg.num_workers)),
            pin_memory=bool(cfg.CUDA),
        )
        self.memory = memory or PCMemory(
            memory_dim=int(pc_cfg.memory_dim),
            value_dim=int(pc_cfg.value_dim),
            geometry_dim=int(pc_cfg.geometry_dim),
            storage_dtype=torch.float16,
            config=pc_cfg,
        )

        self.current_epoch = 1
        self.save_dir = Path(cfg.save_dir)
        if is_main_process():
            self.save_dir.mkdir(parents=True, exist_ok=True)
        synchronize()
        if resume_path:
            checkpoint = load_training_resume(
                resume_path,
                model=self.core_model.student,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                ema_model=self.core_model.teacher,
                restore_rng=True,
            )
            self.current_epoch = int(checkpoint["epoch"]) + 1
            self._freeze_teacher()

    def _validate_model_contract(self):
        if getattr(self.core_model, "pc_cfg", None) is None:
            raise ValueError("TSModel must be constructed with DinoPCHBMConfig")
        if getattr(self.core_model.student, "pc_hbm", None) is None:
            raise ValueError("Student Decoder has no PC-HBM engine")
        if getattr(self.core_model.teacher, "pc_hbm", None) is None:
            raise ValueError("Teacher Decoder has no PC-HBM engine")
        student_keys = tuple(dict(self.core_model.student.named_parameters()))
        teacher_keys = tuple(dict(self.core_model.teacher.named_parameters()))
        if student_keys != teacher_keys:
            raise RuntimeError("Teacher and Student parameter names/order do not match")
        self._freeze_teacher()

    def _freeze_teacher(self):
        self.core_model.teacher.eval()
        self.core_model.teacher.requires_grad_(False)

    def _distributed_sampler(self, dataset):
        if not self.distributed:
            return None
        return DistributedSampler(
            dataset,
            num_replicas=int(self.cfg.world_size),
            rank=int(self.cfg.rank),
            shuffle=True,
            seed=int(self.cfg.seed),
            drop_last=True,
        )

    @staticmethod
    def _cycle_loader(loader):
        while True:
            yield from loader

    def _decoder_epoch(self, ts_epoch: int) -> int:
        """Continue after Base epoch 30 so mixture uses its terminal schedule."""

        base_end = int(getattr(self.pc_cfg, "mixture_schedule_end_epoch", 30))
        return base_end + int(ts_epoch)

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.amp_enabled,
        )

    def _rebuild_memory(self, producer, *, producer_source: str):
        compat_meta = build_memory_compat_meta(
            self.pc_cfg,
            producer,
            producer_source=producer_source,
        )
        rebuild_memory(
            model=self.core_model,
            memory_decoder=producer,
            memory_loader=self.memory_loader,
            memory=self.memory,
            device=self.device,
            config=self.pc_cfg,
            compat_meta=compat_meta,
            use_amp=self.amp_enabled,
        )
        if not self.memory.is_ready():
            raise RuntimeError("PC-HBM training cannot continue with unready memory")
        synchronize()
        return compat_meta

    @staticmethod
    def _clone_teacher_target_aux(aux: Mapping[str, Any]) -> dict[str, Any]:
        """Keep only confidence inputs and clone them outside inference mode."""

        pc = aux.get("pc_hbm", {}) or {}
        mixture = aux.get("mixture", {}) or {}

        def clone_tensor(value, name):
            if not torch.is_tensor(value):
                raise KeyError(f"Teacher pseudo aux is missing {name}")
            cloned = value.detach().clone()
            if cloned.is_inference():
                raise RuntimeError(f"Teacher target {name} remained an inference tensor")
            return cloned

        return {
            "p_final": clone_tensor(aux.get("p_final"), "p_final"),
            "z_main": clone_tensor(aux.get("z_main"), "z_main"),
            "pc_hbm": {
                "C23_map": clone_tensor(pc.get("C23_map"), "pc_hbm.C23_map"),
                "route_entropy_norm": clone_tensor(
                    pc.get("route_entropy_norm"), "pc_hbm.route_entropy_norm"
                ),
            },
            "mixture": {"pi": clone_tensor(mixture.get("pi"), "mixture.pi")},
        }

    def train_epoch(self):
        epoch = int(self.current_epoch)
        decoder_epoch = self._decoder_epoch(epoch)
        self.model.train()
        self._freeze_teacher()
        if self.labeled_sampler is not None:
            self.labeled_sampler.set_epoch(epoch)
        if self.unlabeled_sampler is not None:
            self.unlabeled_sampler.set_epoch(epoch)

        # Every rank independently traverses the complete deterministic labeled
        # loader.  The resulting CPU-FP16 memory remains read-only this epoch.
        self._rebuild_memory(self.core_model.teacher, producer_source="ema_teacher")
        labeled_iter = self._cycle_loader(self.labeled_train_dl)
        totals = {"loss": 0.0, "labeled": 0.0, "unlabeled": 0.0, "confidence": 0.0}
        steps = 0

        progress = tqdm(
            self.unlabeled_train_dl,
            disable=not is_main_process(),
            desc=f"TS PC-HBM epoch {epoch}",
        )
        for u_imgs in progress:
            _, l_imgs, l_gt, l_image_ids = next(labeled_iter)
            l_imgs = l_imgs.to(self.device, non_blocking=bool(self.cfg.CUDA))
            l_gt = l_gt.to(self.device, non_blocking=bool(self.cfg.CUDA))
            u_imgs = u_imgs.to(self.device, non_blocking=bool(self.cfg.CUDA))
            self.optimizer.zero_grad(set_to_none=True)

            # 1) Labeled Student full pass and its synchronized backward.
            with self._autocast():
                l_features = self.core_model.extract_features(l_imgs)
                l_outputs, l_aux = self.model(
                    branch="student_labeled",
                    features=l_features,
                    memory=self.memory,
                    epoch=decoder_epoch,
                    query_image_ids=list(l_image_ids),
                )
                l_loss, l_log = pc_hbm_labeled_loss(
                    l_outputs,
                    l_aux,
                    l_gt,
                    decoder_epoch,
                    self.pc_cfg,
                    pc_mode="full",
                    strict=True,
                )
            self.scaler.scale(l_loss).backward()
            l_loss_value = float(l_loss.detach())
            del l_features, l_outputs, l_aux, l_gt, l_imgs, l_image_ids, l_loss, l_log

            # 2) Unlabeled DINO features, read-only Teacher full inference, and
            # cloning to ordinary tensors only after leaving inference_mode.
            with self._autocast():
                u_features = self.core_model.extract_features(u_imgs)
            with torch.inference_mode():
                with self._autocast():
                    teacher_aux = self.core_model.teacher_pseudo(
                        u_features,
                        self.memory,
                        decoder_epoch,
                    )
            teacher_target_aux = self._clone_teacher_target_aux(teacher_aux)
            del teacher_aux
            pseudo = prepare_pseudo_targets(
                teacher_target_aux,
                self.pc_cfg,
                strict=True,
            )
            del teacher_target_aux

            # 3) Student core deliberately skips P1-PRA/mixture.  This second
            # backward is synchronized normally; no DDP no_sync is used.
            with self._autocast():
                u_outputs, u_aux = self.model(
                    branch="student_unlabeled",
                    features=u_features,
                    memory=self.memory,
                    epoch=decoder_epoch,
                )
                u_loss, u_log = pc_unlabeled_loss(
                    u_outputs,
                    u_aux,
                    pseudo["p_soft"],
                    pseudo["confidence"],
                    epoch,
                    self.pc_cfg,
                )
            self.scaler.scale(u_loss).backward()

            # 4) Exactly one optimizer step, followed by exact-name EMA and
            # buffer copy from Student to frozen Teacher.
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(
                self.core_model.student.parameters(),
                max_norm=float(getattr(self.pc_cfg, "grad_clip_norm", 5.0)),
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            with torch.no_grad():
                update_ema_module(
                    self.core_model.student,
                    self.core_model.teacher,
                    momentum=float(getattr(self.pc_cfg, "ema_momentum", 0.995)),
                )

            u_loss_value = float(u_loss.detach())
            confidence_value = float(u_log["pseudo_conf_mean"])
            total_value = l_loss_value + u_loss_value
            totals["loss"] += total_value
            totals["labeled"] += l_loss_value
            totals["unlabeled"] += u_loss_value
            totals["confidence"] += confidence_value
            steps += 1
            progress.set_postfix(
                loss=f"{total_value:.4f}",
                conf=f"{confidence_value:.3f}",
            )
            del u_features, u_outputs, u_aux, pseudo, u_imgs, u_loss, u_log

        if steps == 0:
            raise RuntimeError("PC-HBM TS epoch completed without optimizer steps")
        means = {
            name: reduce_mean(value / steps, self.device)
            for name, value in totals.items()
        }
        return means

    def _save_epoch(self, epoch: int, metrics: Mapping[str, float]):
        if not is_main_process():
            return
        save_decoder_checkpoint(
            self.save_dir / f"ts_pc_hbm_student_epoch_{epoch}.pth",
            self.core_model.student,
            self.pc_cfg,
            epoch,
            extra={"metrics": dict(metrics), "producer": "student"},
        )
        save_training_resume(
            self.save_dir / "ts_pc_hbm_resume_latest.pth",
            epoch=epoch,
            model=self.core_model.student,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            ema_model=self.core_model.teacher,
            pc_cfg=self.pc_cfg,
            extra={"metrics": dict(metrics)},
        )

    def _export_final_memory(self):
        # Final inference artifacts must have a memory produced by the final
        # Student, not the preceding epoch's EMA Teacher snapshot.
        compat_meta = self._rebuild_memory(
            self.core_model.student,
            producer_source="student_final",
        )
        if is_main_process():
            save_decoder_checkpoint(
                self.save_dir / "ts_pc_hbm_student_final.pth",
                self.core_model.student,
                self.pc_cfg,
                int(self.cfg.epochs),
                extra={"producer": "student_final"},
            )
            save_memory_checkpoint(
                self.save_dir / "ts_pc_hbm_memory_final.pth",
                self.memory,
                compat_meta=compat_meta,
            )

    def train(self):
        if self.current_epoch > int(self.cfg.epochs):
            raise ValueError(
                f"Resume epoch {self.current_epoch} exceeds configured epochs {self.cfg.epochs}"
            )
        for epoch in range(self.current_epoch, int(self.cfg.epochs) + 1):
            self.current_epoch = epoch
            metrics = self.train_epoch()
            self.scheduler.step()
            self._save_epoch(epoch, metrics)
            if is_main_process():
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f">>> TS PC-HBM epoch {epoch}/{self.cfg.epochs}: "
                    f"loss={metrics['loss']:.6f}, confidence={metrics['confidence']:.4f}, "
                    f"lr={lr:.3e}"
                )
            synchronize()
        self._export_final_memory()
        synchronize()


# Concise compatibility name for callers that follow the existing trainer API.
Trainer = PCHBMPseudoTrainer


__all__ = ["PCHBMPseudoTrainer", "Trainer"]
