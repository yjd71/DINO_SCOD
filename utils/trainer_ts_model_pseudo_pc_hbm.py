"""Teacher-only pseudo-label trainer for PC-HBM-Lite."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from Model.PC_HBM.memory import PCMemory
from Model.PC_HBM.training import (
    DIAGNOSTIC_NAMES,
    base_structure_loss,
    collect_pc_diagnostics,
    pc_unlabeled_loss,
    prepare_pseudo_targets,
)
from utils.checkpoint_pc_hbm import (
    build_artifact_metadata,
    CANONICAL_LABELED_SPLIT_COUNT,
    compute_labeled_split_fingerprint,
    load_training_resume,
    read_artifact_metadata,
    save_decoder_checkpoint,
    save_training_resume,
    state_dict_fingerprint,
    validate_canonical_labeled_indices_pt,
    validate_canonical_labeled_split_fingerprint,
    validate_artifact_metadata,
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
from utils.logging_utils import current_time
from utils.pc_memory_runner import (
    build_memory_compat_meta,
    module_fingerprint,
    rebuild_memory,
)


def validate_teacher_enhancer_checkpoint(
    source,
    labeled_split_fingerprint: str,
) -> dict[str, Any]:
    """Validate the frozen Lite Teacher identity before model construction."""

    validate_canonical_labeled_split_fingerprint(
        labeled_split_fingerprint
    )
    return validate_artifact_metadata(
        source,
        {
            "training_design": ("two_stage", "teacher_only"),
            "artifact_role": "teacher_enhancer",
            "labeled_split_fingerprint": str(labeled_split_fingerprint),
            "pc_frozen": True,
        },
    )


class PCHBMPseudoTrainer:
    """Train a raw Student while a frozen Lite enhancer produces soft targets."""

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
        self.training_design = str(
            getattr(cfg, "pc_training_design", "teacher_only")
        )
        if self.training_design != "teacher_only":
            raise ValueError("PC-HBM-Lite TS supports only teacher_only")
        configure = getattr(pc_cfg, "configure_training_design", None)
        if not callable(configure):
            raise RuntimeError("pc_cfg.configure_training_design() is required")
        configure("teacher_only")

        self.device = torch.device(cfg.device)
        self.distributed = bool(getattr(cfg, "distributed", False))
        self.core_model = unwrap_model(model)
        self._validate_model_contract()
        if int(cfg.l_batch_size) != 32 or int(cfg.u_batch_size) != 32:
            raise ValueError(
                "PC-HBM-Lite TS requires labeled and unlabeled batches of 32"
            )

        parameters = [
            parameter
            for parameter in self.core_model.student.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError("Raw Student has no trainable parameters")
        self.optimizer = optim.Adam(
            parameters,
            lr=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
        )
        self.scheduler = scheduler or optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, int(cfg.epochs)),
            eta_min=float(cfg.min_lr),
        )
        self.amp_enabled = bool(
            getattr(pc_cfg, "use_amp", True)
            and not bool(getattr(self.cfg, "pc_force_no_amp", False))
            and self.device.type == "cuda"
        )
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=self.amp_enabled
        )

        indices_path = getattr(cfg, "train_labeled_indices_pt", None)
        if not indices_path:
            raise ValueError(
                "teacher_only TS requires train_labeled_indices_pt"
            )
        self.labeled_train_set = PCLabeledTrainDataset(
            l_image_root=cfg.train_imgs,
            l_gt_root=cfg.train_masks,
            l_txt_root=cfg.train_sample_txt,
            l_train_size=cfg.l_train_size,
            labeled_indices_pt=indices_path,
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
            labeled_indices_pt=indices_path,
        )
        if len(self.labeled_train_set) < 32:
            raise ValueError("Labeled split is smaller than batch size 32")
        if len(self.unlabeled_train_set) < 32:
            raise ValueError("Unlabeled split is smaller than batch size 32")

        self.labeled_sampler = self._distributed_sampler(
            self.labeled_train_set
        )
        self.unlabeled_sampler = self._distributed_sampler(
            self.unlabeled_train_set
        )
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
        self.memory_loader = build_labeled_memory_loader(
            l_image_root=cfg.train_imgs,
            l_gt_root=cfg.train_masks,
            l_txt_root=cfg.train_sample_txt,
            l_train_size=cfg.l_train_size,
            labeled_indices_pt=indices_path,
            batch_size=int(getattr(cfg, "memory_batch_size", 16)),
            num_workers=int(
                getattr(cfg, "memory_num_workers", cfg.num_workers)
            ),
            pin_memory=bool(cfg.CUDA),
        )
        self.memory = memory or PCMemory(config=pc_cfg)

        split_fingerprint = compute_labeled_split_fingerprint(
            self.labeled_train_set.sample_keys
        )
        indices_fingerprint = validate_canonical_labeled_indices_pt(
            indices_path
        )
        if len(self.labeled_train_set.sample_keys) != (
            CANONICAL_LABELED_SPLIT_COUNT
        ):
            raise RuntimeError(
                "TS labeled loader must contain exactly "
                f"{CANONICAL_LABELED_SPLIT_COUNT} samples"
            )
        if split_fingerprint != indices_fingerprint:
            raise RuntimeError(
                "Labeled dataset and indices file fingerprints differ"
            )
        validate_canonical_labeled_split_fingerprint(split_fingerprint)
        prevalidated_split = getattr(
            cfg, "labeled_split_fingerprint", split_fingerprint
        )
        if str(prevalidated_split) != split_fingerprint:
            raise RuntimeError(
                "CLI-prevalidated and dataset labeled split fingerprints differ"
            )
        cfg.labeled_split_fingerprint = split_fingerprint

        teacher_checkpoint = getattr(cfg, "teacher_pc_checkpoint", None)
        if not teacher_checkpoint:
            raise ValueError("teacher_pc_checkpoint is required")
        teacher_metadata = validate_teacher_enhancer_checkpoint(
            teacher_checkpoint,
            split_fingerprint,
        )
        cfg.baseline_fingerprint = teacher_metadata[
            "baseline_fingerprint"
        ]
        teacher_legacy_state = {
            name: value
            for name, value in self.core_model.teacher.state_dict().items()
            if not name.startswith("pc_hbm.")
        }
        if state_dict_fingerprint(teacher_legacy_state) != str(
            cfg.baseline_fingerprint
        ):
            raise RuntimeError(
                "Teacher artifact metadata does not match its legacy Decoder state"
            )
        student_checkpoint = getattr(cfg, "student_checkpoint", None)
        if student_checkpoint:
            validate_artifact_metadata(
                student_checkpoint,
                {
                    "training_design": "teacher_only",
                    "artifact_role": "student_raw",
                    "labeled_split_fingerprint": split_fingerprint,
                    "baseline_fingerprint": cfg.baseline_fingerprint,
                    "pc_frozen": True,
                },
            )
        raw_state = {
            name: value
            for name, value in self.core_model.student.state_dict().items()
            if not name.startswith("pc_hbm.")
        }
        if state_dict_fingerprint(raw_state) != str(
            cfg.baseline_fingerprint
        ):
            raise RuntimeError(
                "Raw Student baseline does not match the Teacher artifact"
            )

        self.save_dir = Path(cfg.save_dir)
        if is_main_process():
            self.save_dir.mkdir(parents=True, exist_ok=True)
        synchronize()
        self.current_epoch = 1
        if resume_path is not None:
            self._resume(resume_path)
        self._freeze_teacher()
        self._teacher_pc_fingerprint = module_fingerprint(
            self.core_model.teacher.pc_hbm
        )

    def _validate_model_contract(self):
        if getattr(self.core_model, "training_design", None) != "teacher_only":
            raise RuntimeError("TSModel must use teacher_only")
        teacher_pc = getattr(self.core_model.teacher, "pc_hbm", None)
        if teacher_pc is None:
            raise RuntimeError("Teacher must contain PC-HBM-Lite")
        if getattr(self.core_model.student, "pc_hbm", None) is not None:
            raise RuntimeError("Student must be the raw legacy Decoder")

    def _freeze_teacher(self):
        self.core_model.teacher.requires_grad_(False).eval()

    def _validate_teacher_pc_contract(self):
        current = module_fingerprint(self.core_model.teacher.pc_hbm)
        if current != self._teacher_pc_fingerprint:
            raise RuntimeError(
                "Teacher PC-HBM-Lite parameters changed during TS training"
            )

    def _distributed_sampler(self, dataset):
        if not self.distributed:
            return None
        return DistributedSampler(
            dataset,
            shuffle=True,
            drop_last=True,
        )

    @staticmethod
    def _cycle_loader(loader):
        while True:
            yield from loader

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.amp_enabled,
        )

    def _rebuild_memory(self):
        compat_meta = build_memory_compat_meta(
            self.pc_cfg,
            self.core_model.teacher,
        )
        rebuild_memory(
            model=self.core_model,
            memory_decoder=self.core_model.teacher,
            memory_loader=self.memory_loader,
            memory=self.memory,
            device=self.device,
            config=self.pc_cfg,
            compat_meta=compat_meta,
            use_amp=self.amp_enabled,
        )
        if not self.memory.is_ready():
            raise RuntimeError("Teacher memory rebuild produced an unready memory")
        compatibility = self.memory.validate_compat(
            compat_meta,
            require_producer_match=True,
        )
        compatible = (
            bool(compatibility[0])
            if isinstance(compatibility, tuple)
            else bool(compatibility)
        )
        if not compatible:
            raise RuntimeError("Teacher memory is incompatible after rebuild")
        synchronize()

    @staticmethod
    def _clone_teacher_target_aux(
        aux: Mapping[str, Any],
    ) -> dict[str, Any]:
        pc = aux.get("pc_hbm")
        distill = aux.get("distill_features")
        if not isinstance(pc, Mapping) or not isinstance(distill, Mapping):
            raise KeyError("Teacher Lite aux mappings are incomplete")
        required = {
            "p_final": aux.get("p_final"),
            "query_mask_map": pc.get("query_mask_map"),
            "memory_confidence_map": pc.get("memory_confidence_map"),
            "p3_corr": distill.get("p3_corr"),
        }
        missing = [
            name for name, value in required.items() if not torch.is_tensor(value)
        ]
        if missing:
            raise KeyError(f"Teacher targets are missing tensors: {missing}")
        return {
            "p_final": required["p_final"].detach().clone(),
            "pc_active": True,
            "fallback_reason": None,
            "forward_mode": "teacher_pseudo",
            "pc_hbm": {
                "query_mask_map": required["query_mask_map"].detach().clone(),
                "memory_confidence_map": required[
                    "memory_confidence_map"
                ].detach().clone(),
            },
            "distill_features": {
                "p3_corr": required["p3_corr"].detach().clone(),
            },
        }

    def train_epoch(self):
        epoch = int(self.current_epoch)
        self.model.train()
        self._freeze_teacher()
        if self.labeled_sampler is not None:
            self.labeled_sampler.set_epoch(epoch)
        if self.unlabeled_sampler is not None:
            self.unlabeled_sampler.set_epoch(epoch)
        self._rebuild_memory()

        labeled_iter = self._cycle_loader(self.labeled_train_dl)
        totals: dict[str, float] = {
            "loss": 0.0,
            "labeled": 0.0,
            "unlabeled": 0.0,
            "confidence": 0.0,
            "coverage": 0.0,
            "p3_distill": 0.0,
        }
        totals.update({name: 0.0 for name in DIAGNOSTIC_NAMES})
        steps = 0
        progress = tqdm(
            self.unlabeled_train_dl,
            disable=not is_main_process(),
            desc=f"TS PC-HBM-Lite epoch {epoch}",
        )
        for unlabeled_images in progress:
            _, labeled_images, labeled_gt, _ = next(labeled_iter)
            labeled_images = labeled_images.to(
                self.device, non_blocking=bool(self.cfg.CUDA)
            )
            labeled_gt = labeled_gt.to(
                self.device, non_blocking=bool(self.cfg.CUDA)
            )
            unlabeled_images = unlabeled_images.to(
                self.device, non_blocking=bool(self.cfg.CUDA)
            )
            self.optimizer.zero_grad(set_to_none=True)

            with self._autocast():
                labeled_features = self.core_model.extract_features(
                    labeled_images
                )
                labeled_outputs, _ = self.model(
                    branch="student_labeled",
                    features=labeled_features,
                )
                labeled_loss = base_structure_loss(
                    labeled_outputs, labeled_gt
                )

                unlabeled_features = self.core_model.extract_features(
                    unlabeled_images
                )
            with torch.inference_mode(), self._autocast():
                teacher_aux = self.core_model.teacher_pseudo(
                    unlabeled_features,
                    self.memory,
                    epoch,
                )
            cloned_aux = self._clone_teacher_target_aux(teacher_aux)
            pseudo = prepare_pseudo_targets(
                cloned_aux,
            )
            diagnostics = collect_pc_diagnostics(
                teacher_aux,
                pseudo_confidence=pseudo["confidence"],
            )
            with self._autocast():
                student_outputs, student_aux = self.model(
                    branch="student_unlabeled",
                    features=unlabeled_features,
                )
                unlabeled_loss, unlabeled_log = pc_unlabeled_loss(
                    student_outputs,
                    student_aux,
                    pseudo["p_soft"],
                    pseudo["confidence"],
                    self.pc_cfg,
                    teacher_features={"p3_corr": pseudo["p3_corr"]},
                )
                total_loss = labeled_loss + unlabeled_loss
            if not bool(torch.isfinite(total_loss.detach())):
                raise FloatingPointError(
                    f"Non-finite TS loss at epoch={epoch}, step={steps + 1}"
                )
            self.scaler.scale(total_loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(
                self.core_model.student.parameters(),
                float(getattr(self.pc_cfg, "grad_clip_norm", 5.0)),
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            with torch.no_grad():
                self.core_model.update_teacher(
                    momentum=float(
                        getattr(self.pc_cfg, "ema_momentum", 0.995)
                    ),
                )
            self._validate_teacher_pc_contract()

            totals["loss"] += float(total_loss.detach())
            totals["labeled"] += float(labeled_loss.detach())
            totals["unlabeled"] += float(unlabeled_loss.detach())
            totals["confidence"] += float(
                unlabeled_log["pseudo_conf_mean"]
            )
            totals["coverage"] += float(
                unlabeled_log["pseudo_coverage"]
            )
            totals["p3_distill"] += float(
                unlabeled_log["L_u_feat_p3"]
            )
            for name in DIAGNOSTIC_NAMES:
                totals[name] += float(diagnostics[name])
            steps += 1

        if steps == 0:
            raise RuntimeError("Unlabeled loader produced no full batches")
        metrics = {
            name: reduce_mean(value / steps, self.device)
            for name, value in totals.items()
        }
        self.scheduler.step()
        self.current_epoch += 1
        return metrics

    def _artifact_metadata(self, artifact_role: str) -> dict[str, Any]:
        return build_artifact_metadata(
            training_design="teacher_only",
            artifact_role=artifact_role,
            labeled_split_fingerprint=str(
                self.cfg.labeled_split_fingerprint
            ),
            baseline_fingerprint=str(self.cfg.baseline_fingerprint),
            pc_frozen=True,
        )

    def _save_epoch(self, epoch: int, metrics: Mapping[str, float]):
        if not is_main_process():
            return
        save_decoder_checkpoint(
            self.save_dir / f"student_raw_epoch_{epoch}.pth",
            self.core_model.student,
            self.pc_cfg,
            epoch,
            artifact_meta=self._artifact_metadata("student_raw"),
            extra={"metrics": dict(metrics), "producer": "student_raw"},
        )
        save_training_resume(
            self.save_dir / "ts_pc_hbm_lite_resume_latest.pth",
            epoch=epoch,
            model=self.core_model.student,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            ema_model=self.core_model.teacher,
            pc_cfg=self.pc_cfg,
            artifact_meta=self._artifact_metadata("resume"),
            extra={"metrics": dict(metrics)},
        )

    def _resume(self, path):
        checkpoint = torch.load(
            path, map_location="cpu", weights_only=False
        )
        if not isinstance(checkpoint, Mapping):
            raise TypeError("TS resume checkpoint must be a mapping")
        saved_config = checkpoint.get("pc_cfg")
        current_config = vars(self.pc_cfg)
        if not isinstance(saved_config, Mapping) or dict(saved_config) != dict(
            current_config
        ):
            raise RuntimeError("TS resume PC-HBM-Lite config mismatch")
        validate_artifact_metadata(
            checkpoint,
            {
                "training_design": "teacher_only",
                "artifact_role": "resume",
                "labeled_split_fingerprint": self.cfg.labeled_split_fingerprint,
                "baseline_fingerprint": self.cfg.baseline_fingerprint,
                "pc_frozen": True,
            },
        )
        completed = int(checkpoint.get("epoch", 0))
        if completed < 1 or completed > int(self.cfg.epochs):
            raise RuntimeError(f"Invalid TS resume epoch: {completed}")
        load_training_resume(
            checkpoint,
            model=self.core_model.student,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            ema_model=self.core_model.teacher,
            pc_cfg=self.pc_cfg,
            restore_rng=True,
        )
        self.current_epoch = completed + 1

    def _export_final_student(self):
        if not is_main_process():
            return
        save_decoder_checkpoint(
            self.save_dir / "student_raw.pth",
            self.core_model.student,
            self.pc_cfg,
            int(self.cfg.epochs),
            artifact_meta=self._artifact_metadata("student_raw"),
            extra={"producer": "student_raw_final"},
        )

    def train(self):
        if is_main_process():
            print(f"{current_time()} >>> TS PC-HBM-Lite training starts")
        while self.current_epoch <= int(self.cfg.epochs):
            epoch = self.current_epoch
            metrics = self.train_epoch()
            self._save_epoch(epoch, metrics)
            if is_main_process():
                print(
                    f"{current_time()} [TS PC-HBM-Lite] epoch={epoch} "
                    + " ".join(
                        f"{name}={value:.6f}"
                        for name, value in sorted(metrics.items())
                    )
                )
            synchronize()
        self._export_final_student()
        if is_main_process():
            print(f"{current_time()} <<< TS PC-HBM-Lite training finished")


__all__ = [
    "PCHBMPseudoTrainer",
    "validate_teacher_enhancer_checkpoint",
]
