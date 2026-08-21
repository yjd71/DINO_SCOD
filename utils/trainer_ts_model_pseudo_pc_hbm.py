from __future__ import annotations

from collections.abc import Mapping, Sequence
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
    collect_pc_diagnostics,
    pc_unlabeled_loss,
    prepare_pseudo_targets,
    ts_labeled_pc_loss,
)
from utils.checkpoint_pc_hbm import (
    build_artifact_metadata,
    load_training_resume,
    save_decoder_checkpoint,
    save_memory_checkpoint,
    save_training_resume,
    state_dict_fingerprint,
    validate_artifact_metadata,
    validate_labeled_sample_keys,
    validate_labeled_split_source,
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
from utils.pc_memory_runner import build_memory_compat_meta, rebuild_memory


class PCHBMPseudoTrainer:
    """Train a Student-owned PC-HBM with EMA pseudo-label supervision."""

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
            getattr(cfg, "pc_training_design", "student_joint")
        )
        if self.training_design != "student_joint":
            raise ValueError("PC-HBM-Lite TS supports only student_joint")
        if pc_cfg is None:
            raise ValueError("pc_cfg is required for student_joint TS")

        self.device = torch.device(cfg.device)
        self.distributed = bool(getattr(cfg, "distributed", False))
        self.core_model = unwrap_model(model)
        self._validate_model_contract()
        if int(cfg.l_batch_size) != 32 or int(cfg.u_batch_size) != 32:
            raise ValueError("PC-HBM-Lite TS requires labeled and unlabeled batches of 32")

        parameters = [
            parameter
            for parameter in self.core_model.student.parameters()
            if parameter.requires_grad
        ]
        if not parameters:
            raise RuntimeError("Student has no trainable parameters")
        self.optimizer = optim.Adam(
            parameters,
            lr=float(cfg.learning_rate),
            weight_decay=float(cfg.weight_decay),
        )
        self._validate_optimizer_contract()
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
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        indices_path = getattr(cfg, "train_labeled_indices_pt", None)
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
        self.memory_loader = build_labeled_memory_loader(
            l_image_root=cfg.train_imgs,
            l_gt_root=cfg.train_masks,
            l_txt_root=cfg.train_sample_txt,
            l_train_size=cfg.l_train_size,
            labeled_indices_pt=indices_path,
            batch_size=int(getattr(cfg, "memory_batch_size", 16)),
            num_workers=int(getattr(cfg, "memory_num_workers", cfg.num_workers)),
            pin_memory=bool(cfg.CUDA),
        )
        self.memory = memory or PCMemory(config=pc_cfg)

        split_identity = validate_labeled_sample_keys(self.labeled_train_set.sample_keys)
        source_identity = validate_labeled_split_source(
            indices_path, getattr(cfg, "train_sample_txt", None)
        )
        if split_identity != source_identity:
            raise RuntimeError("Labeled dataset and selection source split identities differ")
        if (
            str(getattr(cfg, "labeled_split_fingerprint", split_identity.fingerprint))
            != split_identity.fingerprint
            or int(getattr(cfg, "labeled_split_count", split_identity.count))
            != split_identity.count
        ):
            raise RuntimeError("CLI-prevalidated and dataset labeled split identities differ")
        cfg.labeled_split_fingerprint = split_identity.fingerprint
        cfg.labeled_split_count = split_identity.count

        self.base_student_fingerprint = str(cfg.baseline_fingerprint)
        self.current_epoch = 1
        self.memory_compat_meta: dict[str, Any] | None = None
        self.save_dir = Path(cfg.save_dir)
        if is_main_process():
            self.save_dir.mkdir(parents=True, exist_ok=True)
        synchronize()
        if resume_path:
            self._resume(resume_path)
        else:
            self.core_model.sync_readonly_pc()

    def _validate_model_contract(self):
        if getattr(self.core_model, "training_design", None) != "student_joint":
            raise RuntimeError("TSModel must use student_joint")
        if getattr(self.core_model.student, "pc_hbm", None) is None:
            raise RuntimeError("Student must own the trainable PC-HBM-Lite engine")
        if getattr(self.core_model.teacher, "pc_hbm", None) is not None:
            raise RuntimeError("Teacher must not register its own PC-HBM engine")
        readonly = getattr(self.core_model, "pc_hbm_readonly", None)
        if readonly is None:
            raise RuntimeError("TSModel must expose pc_hbm_readonly")
        if any(parameter.requires_grad for parameter in self.core_model.teacher.parameters()):
            raise RuntimeError("Teacher parameters must be frozen")
        if any(parameter.requires_grad for parameter in readonly.parameters()):
            raise RuntimeError("Readonly PC-HBM parameters must be frozen")
        self.core_model.assert_readonly_pc_synced()

    def _validate_optimizer_contract(self):
        optimizer_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        }
        student_ids = {
            id(parameter)
            for parameter in self.core_model.student.parameters()
            if parameter.requires_grad
        }
        forbidden_ids = {
            id(parameter)
            for module in (self.core_model.teacher, self.core_model.pc_hbm_readonly)
            for parameter in module.parameters()
        }
        if optimizer_ids != student_ids:
            raise RuntimeError("TS optimizer must contain every trainable Student parameter")
        if optimizer_ids.intersection(forbidden_ids):
            raise RuntimeError("Teacher or readonly PC-HBM leaked into the optimizer")

    def _distributed_sampler(self, dataset):
        if not self.distributed:
            return None
        return DistributedSampler(
            dataset,
            shuffle=True,
            seed=int(self.cfg.seed),
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

    def _rebuild_memory(self, *, force=False):
        interval = int(getattr(self.pc_cfg, "memory_refresh_interval_epochs", 1))
        should_rebuild = (
            force
            or not self.memory.is_ready()
            or (int(self.current_epoch) - 1) % interval == 0
        )
        if not should_rebuild:
            return
        compat_meta = build_memory_compat_meta(self.pc_cfg, self.core_model.student)
        rebuild_memory(
            model=self.core_model,
            memory_decoder=self.core_model.student,
            memory_loader=self.memory_loader,
            memory=self.memory,
            device=self.device,
            config=self.pc_cfg,
            compat_meta=compat_meta,
            entry_builder=self.core_model.student.pc_hbm.build_memory_entries,
            use_amp=self.amp_enabled,
            expected_split_count=int(self.cfg.labeled_split_count),
            expected_split_fingerprint=str(self.cfg.labeled_split_fingerprint),
        )
        compatibility = self.memory.validate_compat(
            compat_meta, require_producer_match=True
        )
        if not bool(compatibility):
            raise RuntimeError(
                f"Labeled Student memory is incompatible after rebuild: {compatibility.reason}"
            )
        self.memory_compat_meta = compat_meta
        synchronize()

    def train_epoch(self):
        epoch = int(self.current_epoch)
        self.model.train()
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
            "readonly_pc_sync_max_abs_diff": 0.0,
        }
        totals.update({name: 0.0 for name in DIAGNOSTIC_NAMES})
        steps = 0
        progress = tqdm(
            self.unlabeled_train_dl,
            disable=not is_main_process(),
            desc=f"TS Student-joint epoch {epoch}",
        )
        for unlabeled_images in progress:
            labeled_images, labeled_gt, labeled_ids = self._unpack_labeled_batch(
                next(labeled_iter)
            )
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
                labeled_features = self.core_model.extract_features(labeled_images)
                unlabeled_features = self.core_model.extract_features(unlabeled_images)
            with torch.inference_mode(), self._autocast():
                teacher_aux = self.core_model.teacher_pseudo(
                    unlabeled_features, self.memory, epoch
                )
                pseudo = prepare_pseudo_targets(teacher_aux)
            diagnostics = collect_pc_diagnostics(
                teacher_aux, pseudo_confidence=pseudo["confidence"]
            )

            with self._autocast():
                joint = self.model(
                    branch="student_joint",
                    labeled_features=labeled_features,
                    unlabeled_features=unlabeled_features,
                    memory=self.memory,
                    epoch=epoch,
                    labeled_image_ids=labeled_ids,
                )
                labeled_outputs, labeled_aux = joint["labeled"]
                unlabeled_outputs, unlabeled_aux = joint["unlabeled"]
                labeled_loss, labeled_log = ts_labeled_pc_loss(
                    labeled_outputs, labeled_aux, labeled_gt, self.pc_cfg
                )
                unlabeled_loss, unlabeled_log = pc_unlabeled_loss(
                    unlabeled_outputs,
                    unlabeled_aux,
                    pseudo["p_soft"],
                    pseudo["confidence"],
                    self.pc_cfg,
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
            self.core_model.update_teacher(
                momentum=float(getattr(self.pc_cfg, "ema_momentum", 0.995))
            )
            self.core_model.sync_readonly_pc()
            sync_diff = self.core_model.readonly_pc_sync_max_abs_diff()

            totals["loss"] += float(total_loss.detach())
            totals["labeled"] += float(labeled_loss.detach())
            totals["unlabeled"] += float(unlabeled_loss.detach())
            totals["confidence"] += float(unlabeled_log["pseudo_conf_mean"])
            totals["coverage"] += float(unlabeled_log["pseudo_coverage"])
            totals["readonly_pc_sync_max_abs_diff"] += sync_diff
            for name in DIAGNOSTIC_NAMES:
                totals[name] += float(diagnostics[name])
            totals.setdefault("L_l_seg", 0.0)
            totals.setdefault("L_l_pair", 0.0)
            totals.setdefault("L_u_pseudo", 0.0)
            totals["L_l_seg"] += float(labeled_log["L_l_seg"])
            totals["L_l_pair"] += float(labeled_log["L_l_pair"])
            totals["L_u_pseudo"] += float(unlabeled_log["L_u_pseudo"])
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
        baseline = self.base_student_fingerprint
        if artifact_role in {"ts_student", "ts_student_memory"}:
            baseline = state_dict_fingerprint(
                {
                    name: value
                    for name, value in self.core_model.student.state_dict().items()
                    if not name.startswith("pc_hbm.")
                }
            )
        return build_artifact_metadata(
            training_design="student_joint",
            artifact_role=artifact_role,
            labeled_split_fingerprint=str(self.cfg.labeled_split_fingerprint),
            baseline_fingerprint=baseline,
            pc_frozen=False,
        )

    def _save_epoch(self, epoch: int, metrics: Mapping[str, float]):
        if not is_main_process():
            return
        save_decoder_checkpoint(
            self.save_dir / f"ts_student_epoch_{epoch:02d}.pth",
            self.core_model.student,
            self.pc_cfg,
            epoch,
            artifact_meta=self._artifact_metadata("ts_student"),
            extra={"metrics": dict(metrics), "producer": "labeled_student"},
        )
        save_training_resume(
            self.save_dir / "ts_resume_latest.pth",
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
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise TypeError("TS resume checkpoint must be a mapping")
        validate_artifact_metadata(
            checkpoint,
            {
                "training_design": "student_joint",
                "artifact_role": "resume",
                "labeled_split_fingerprint": self.cfg.labeled_split_fingerprint,
                "baseline_fingerprint": self.base_student_fingerprint,
                "pc_frozen": False,
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
        self.core_model.sync_readonly_pc()
        self.current_epoch = completed + 1

    def _export_final_student(self):
        self._rebuild_memory(force=True)
        if not is_main_process():
            return
        final_epoch = max(0, self.current_epoch - 1)
        compat_meta = build_memory_compat_meta(self.pc_cfg, self.core_model.student)
        save_decoder_checkpoint(
            self.save_dir / "ts_student_final.pth",
            self.core_model.student,
            self.pc_cfg,
            final_epoch,
            artifact_meta=self._artifact_metadata("ts_student"),
            extra={"producer": "labeled_student_final"},
        )
        save_memory_checkpoint(
            self.save_dir / "ts_student_memory_final.pth",
            self.memory,
            compat_meta=compat_meta,
            artifact_meta=self._artifact_metadata("ts_student_memory"),
        )

    def train(self):
        if is_main_process():
            print(f"{current_time()} >>> TS Student-joint training starts")
        while self.current_epoch <= int(self.cfg.epochs):
            epoch = self.current_epoch
            metrics = self.train_epoch()
            self._save_epoch(epoch, metrics)
            if is_main_process():
                print(
                    f"{current_time()} [TS Student-joint] epoch={epoch} "
                    + " ".join(
                        f"{name}={value:.6f}"
                        for name, value in sorted(metrics.items())
                    )
                )
            synchronize()
        self._export_final_student()
        if is_main_process():
            print(f"{current_time()} <<< TS Student-joint training finished")

    @staticmethod
    def _unpack_labeled_batch(batch) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        if isinstance(batch, Mapping):
            images = batch.get("images", batch.get("image"))
            gt = batch.get("gt", batch.get("masks"))
            image_ids = batch.get("image_ids", batch.get("sample_keys"))
        elif isinstance(batch, (tuple, list)) and len(batch) == 4:
            _, images, gt, image_ids = batch
        else:
            raise TypeError("TS labeled batches must contain images, GT and stable IDs")
        if not torch.is_tensor(images) or not torch.is_tensor(gt):
            raise TypeError("TS labeled images and GT must be tensors")
        if isinstance(image_ids, str):
            image_ids = [image_ids]
        elif isinstance(image_ids, Sequence):
            image_ids = [str(value) for value in image_ids]
        else:
            raise TypeError("TS labeled image IDs must be a string sequence")
        if len(image_ids) != images.shape[0]:
            raise ValueError("TS labeled image ID count differs from batch size")
        return images, gt, image_ids


__all__ = ["PCHBMPseudoTrainer", "current_time"]
