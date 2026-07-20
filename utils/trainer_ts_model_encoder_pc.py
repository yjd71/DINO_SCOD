"""Teacher/Student training for the isolated encoder-side PC-HBM v3 profile."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from configs.pc_hbm_dino_config import EncoderPCHBMConfig
from Model.PC_HBM.encoder import EncoderPCMemory
from Model.PC_HBM.encoder.teacher_pseudo_refiner import (
    teacher_pseudo_refiner_labeled_loss,
)
from Model.PC_HBM.training import (
    EncoderPCStage,
    build_encoder_pc_optimizer,
    configure_encoder_pc_stage,
    encoder_pc_labeled_loss,
    encoder_pc_unlabeled_loss,
    prepare_encoder_pc_pseudo_targets,
)
from utils.checkpoint_pc_hbm import (
    capture_rng_state,
    compute_labeled_split_fingerprint,
    load_encoder_pc_training_resume,
    save_encoder_pc_checkpoint,
    save_encoder_pc_training_resume,
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
from utils.pc_memory_runner import (
    module_fingerprint,
    rebuild_encoder_memory,
)


ENCODER_PC_TS_TRAINING_DESIGN = "teacher_student"


class EncoderPCTSTrainer:
    """Train full Student Adapter/Decoder/Refiner from an EMA Teacher.

    The labeled and unlabeled graphs are forwarded and backpropagated in
    sequence.  Exactly one optimizer/scaler step follows both backwards, then
    all three Teacher modules are updated by exact-name EMA.
    """

    def __init__(
        self,
        model,
        cfg,
        pc_cfg: EncoderPCHBMConfig,
        *,
        memory: EncoderPCMemory | None = None,
        scheduler=None,
        resume_path: str | Path | None = None,
        labeled_loader=None,
        unlabeled_loader=None,
        memory_loader=None,
        optimizer=None,
        memory_rebuild_fn: Callable[..., Any] = rebuild_encoder_memory,
    ) -> None:
        self.model = model
        self.core_model = unwrap_model(model)
        self.cfg = cfg
        if not isinstance(pc_cfg, EncoderPCHBMConfig):
            raise TypeError("EncoderPCTSTrainer requires EncoderPCHBMConfig")
        self.pc_cfg = pc_cfg
        self.training_design = ENCODER_PC_TS_TRAINING_DESIGN
        self.device = torch.device(getattr(cfg, "device", "cpu"))
        self.distributed = bool(getattr(cfg, "distributed", False))
        self._validate_model_contract()

        if int(getattr(cfg, "l_batch_size", 0)) != 32:
            raise ValueError("encoder_pc TS requires physical labeled batch 32")
        if int(getattr(cfg, "u_batch_size", 0)) != 32:
            raise ValueError("encoder_pc TS requires physical unlabeled batch 32")

        self.full_stage = EncoderPCStage.for_epoch(pc_cfg.final_epoch, pc_cfg)
        self.optimizer = optimizer or build_encoder_pc_optimizer(
            self.core_model.student_encoder_pc_hbm,
            self.core_model.student,
            self.core_model.student_pseudo_refiner,
            decoder_warm_started=True,
        )
        self.scheduler = scheduler or CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, int(getattr(cfg, "epochs", 15))),
            eta_min=float(getattr(cfg, "min_lr", 1.0e-6)),
        )
        self.amp_enabled = bool(
            getattr(cfg, "use_amp", True) and self.device.type == "cuda"
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

        self.labeled_sampler = None
        self.unlabeled_sampler = None
        if labeled_loader is None:
            labeled_loader = self._build_labeled_loader()
        if unlabeled_loader is None:
            unlabeled_loader = self._build_unlabeled_loader()
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.memory_loader = (
            memory_loader if memory_loader is not None else self._build_memory_loader()
        )
        if len(self.labeled_loader) == 0 or len(self.unlabeled_loader) == 0:
            raise RuntimeError("encoder_pc TS loaders must contain full batches")

        self.split_state = self._build_split_state()
        self._validate_base_artifact_split()
        self.memory_profile = {
            "schema_version": pc_cfg.memory_schema_version,
            "source": pc_cfg.memory_source,
            "storage_dtype": pc_cfg.memory_storage_dtype,
            "device": pc_cfg.memory_device,
        }
        self.memory = memory or EncoderPCMemory(
            memory_dim=pc_cfg.memory_dim,
            value_dim=pc_cfg.value_dim,
            geometry_dim=pc_cfg.geometry_dim,
            storage_dtype=pc_cfg.memory_storage_dtype,
        )
        self.memory_rebuild_fn = memory_rebuild_fn
        self.current_epoch = 1
        self.save_dir = Path(getattr(cfg, "save_dir", "./results/encoder_pc/ts_model"))
        if is_main_process():
            self.save_dir.mkdir(parents=True, exist_ok=True)
        synchronize()
        self._freeze_teacher()
        if resume_path is not None:
            self.resume(resume_path)

    def _validate_model_contract(self) -> None:
        model = self.core_model
        if not bool(getattr(model, "encoder_pc_profile_v3", False)):
            raise TypeError("TSModel is not configured for encoder_pc v3")
        required = (
            "teacher_encoder_pc_hbm",
            "student_encoder_pc_hbm",
            "teacher",
            "student",
            "teacher_pseudo_refiner",
            "student_pseudo_refiner",
        )
        missing = [name for name in required if not isinstance(getattr(model, name, None), nn.Module)]
        if missing:
            raise AttributeError(f"encoder_pc TS model is missing modules: {missing}")
        for role in ("teacher", "student"):
            decoder = getattr(model, role)
            if getattr(decoder, "pc_hbm", None) is not None:
                raise RuntimeError(f"encoder_pc {role} Decoder must have pc_hbm=None")
            if any(name.startswith("pc_hbm.") for name in decoder.state_dict()):
                raise RuntimeError(f"encoder_pc {role} Decoder contains pc_hbm state")
        for student_name, teacher_name in (
            ("student_encoder_pc_hbm", "teacher_encoder_pc_hbm"),
            ("student", "teacher"),
            ("student_pseudo_refiner", "teacher_pseudo_refiner"),
        ):
            student = getattr(model, student_name)
            teacher = getattr(model, teacher_name)
            if tuple(dict(student.named_parameters())) != tuple(dict(teacher.named_parameters())):
                raise RuntimeError(f"{student_name}/{teacher_name} parameter names differ")
            if tuple(dict(student.named_buffers())) != tuple(dict(teacher.named_buffers())):
                raise RuntimeError(f"{student_name}/{teacher_name} buffer names differ")
        dino = getattr(model, "dino", None)
        if not isinstance(dino, nn.Module):
            raise AttributeError("encoder_pc TS model must expose frozen DINO")
        dino.requires_grad_(False).eval()
        if not isinstance(getattr(model, "encoder_base_artifact_meta", None), Mapping):
            raise RuntimeError("encoder_pc TS was not initialized from a Base v3 artifact")

    def _freeze_teacher(self) -> None:
        self.core_model.dino.requires_grad_(False).eval()
        for module in (
            self.core_model.teacher_encoder_pc_hbm,
            self.core_model.teacher,
            self.core_model.teacher_pseudo_refiner,
        ):
            module.requires_grad_(False).eval()

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

    def _loader_kwargs(self) -> dict[str, Any]:
        workers = int(getattr(self.cfg, "num_workers", 0))
        return {
            "num_workers": workers,
            "pin_memory": bool(getattr(self.cfg, "CUDA", self.device.type == "cuda")),
            "persistent_workers": workers > 0,
            "drop_last": True,
        }

    def _build_labeled_loader(self):
        dataset = PCLabeledTrainDataset(
            l_image_root=self.cfg.train_imgs,
            l_gt_root=self.cfg.train_masks,
            l_txt_root=self.cfg.train_sample_txt,
            l_train_size=self.cfg.l_train_size,
            labeled_indices_pt=getattr(self.cfg, "train_labeled_indices_pt", None),
            rVFlip=True,
            rCrop=True,
            rRotate=False,
            colorEnhance=True,
            rPeper=False,
        )
        if len(dataset) < 32:
            raise ValueError("Labeled set is smaller than the fixed encoder_pc batch 32")
        self.labeled_sampler = self._distributed_sampler(dataset)
        return DataLoader(
            dataset,
            batch_size=32,
            shuffle=self.labeled_sampler is None,
            sampler=self.labeled_sampler,
            **self._loader_kwargs(),
        )

    def _build_unlabeled_loader(self):
        dataset = UnlabeledPseudoTrainDataset(
            u_image_root=self.cfg.train_imgs,
            sampled_txt=self.cfg.train_sample_txt,
            u_train_size=self.cfg.u_train_size,
            labeled_indices_pt=getattr(self.cfg, "train_labeled_indices_pt", None),
        )
        if len(dataset) < 32:
            raise ValueError("Unlabeled set is smaller than the fixed encoder_pc batch 32")
        self.unlabeled_sampler = self._distributed_sampler(dataset)
        return DataLoader(
            dataset,
            batch_size=32,
            shuffle=self.unlabeled_sampler is None,
            sampler=self.unlabeled_sampler,
            **self._loader_kwargs(),
        )

    def _build_memory_loader(self):
        return build_labeled_memory_loader(
            l_image_root=self.cfg.train_imgs,
            l_gt_root=self.cfg.train_masks,
            l_txt_root=self.cfg.train_sample_txt,
            l_train_size=self.cfg.l_train_size,
            labeled_indices_pt=getattr(self.cfg, "train_labeled_indices_pt", None),
            batch_size=int(getattr(self.cfg, "memory_batch_size", 16)),
            num_workers=int(getattr(self.cfg, "memory_num_workers", 0)),
            pin_memory=bool(getattr(self.cfg, "CUDA", self.device.type == "cuda")),
        )

    def _build_split_state(self) -> dict[str, Any]:
        dataset = getattr(self.memory_loader, "dataset", None)
        sample_keys = getattr(dataset, "sample_keys", None)
        if sample_keys:
            keys = [str(item) for item in sample_keys]
        else:
            size = len(dataset) if dataset is not None else len(self.memory_loader)
            keys = [f"@encoder-ts-loader/{index}" for index in range(size)]
        return {
            "fingerprint": compute_labeled_split_fingerprint(keys),
            "sample_count": len(keys),
        }

    def _validate_base_artifact_split(self) -> None:
        metadata = self.core_model.encoder_base_artifact_meta
        saved = metadata.get("split_fingerprint")
        if not isinstance(saved, str) or not saved:
            raise RuntimeError("Base encoder-PC v3 artifact is missing split_fingerprint")
        current = self.split_state["fingerprint"]
        if saved != current:
            raise RuntimeError(
                "Base encoder-PC split fingerprint differs from the TS labeled split"
            )
        saved_dino = metadata.get("dino_weight_fingerprint")
        live_dino = module_fingerprint(self.core_model.dino)
        if not isinstance(saved_dino, str) or not saved_dino:
            raise RuntimeError(
                "Base encoder-PC v3 artifact is missing dino_weight_fingerprint"
            )
        if saved_dino != live_dino:
            raise RuntimeError(
                "Base encoder-PC DINO fingerprint differs from the live frozen DINO"
            )

    @staticmethod
    def _cycle(loader):
        while True:
            yield from loader

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.amp_enabled,
        )

    def _rebuild_memory(self, adapter: nn.Module, *, producer_source: str) -> None:
        self.memory_rebuild_fn(
            self.core_model,
            adapter,
            self.memory_loader,
            self.memory,
            self.device,
            config=self.pc_cfg,
            compat_meta={
                "labeled_split_fingerprint": self.split_state["fingerprint"],
            },
            use_amp=self.amp_enabled,
            producer_source=str(producer_source),
        )
        if not self.memory.is_ready():
            raise RuntimeError("encoder_pc TS memory rebuild did not produce schema v3")
        synchronize()

    def train_epoch(self, epoch: int | None = None) -> dict[str, float]:
        epoch = self.current_epoch if epoch is None else int(epoch)
        if epoch != self.current_epoch:
            raise ValueError(
                f"train_epoch expected epoch {self.current_epoch}, received {epoch}"
            )
        self.model.train()
        configure_encoder_pc_stage(
            self.core_model.student_encoder_pc_hbm,
            self.core_model.student,
            self.core_model.student_pseudo_refiner,
            self.full_stage,
        )
        self._freeze_teacher()
        if self.labeled_sampler is not None:
            self.labeled_sampler.set_epoch(epoch)
        if self.unlabeled_sampler is not None:
            self.unlabeled_sampler.set_epoch(epoch)
        self._rebuild_memory(
            self.core_model.teacher_encoder_pc_hbm,
            producer_source="ema_teacher_adapter",
        )

        totals: dict[str, float] = defaultdict(float)
        steps = 0
        labeled_iter = self._cycle(self.labeled_loader)
        progress = tqdm(
            self.unlabeled_loader,
            disable=not is_main_process(),
            desc=f"TS encoder-PC epoch {epoch}",
        )
        for u_imgs in progress:
            _, l_imgs, l_gt, l_image_ids = next(labeled_iter)
            l_imgs = l_imgs.to(self.device, non_blocking=self.device.type == "cuda")
            l_gt = l_gt.to(self.device, non_blocking=self.device.type == "cuda")
            u_imgs = u_imgs.to(self.device, non_blocking=self.device.type == "cuda")
            self.optimizer.zero_grad(set_to_none=True)

            # Labeled: full core supervision plus a detached refiner graph.
            with self._autocast():
                l_bundle = self.core_model.extract_feature_bundle(l_imgs)
                l_rgb = self.core_model.prepare_rgb(l_imgs)
                l_outputs, l_aux = self.model(
                    branch="student_labeled",
                    features=l_bundle,
                    image_rgb=l_rgb,
                    memory=self.memory,
                    epoch=self.pc_cfg.final_epoch + epoch,
                    query_image_ids=list(l_image_ids),
                )
                l_core_loss, l_terms = encoder_pc_labeled_loss(
                    l_outputs,
                    l_aux,
                    l_gt,
                    self.pc_cfg,
                    self.full_stage,
                )
                l_refiner_loss, refiner_terms = teacher_pseudo_refiner_labeled_loss(
                    l_aux["pseudo_refiner"], l_gt, self.pc_cfg
                )
                l_loss = l_core_loss + l_refiner_loss
            self.scaler.scale(l_loss).backward()
            l_loss_value = float(l_loss.detach())
            for name, value in {**l_terms, **refiner_terms}.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    totals[name] += float(value.detach())
            del l_bundle, l_rgb, l_outputs, l_aux, l_gt, l_imgs, l_loss
            del l_core_loss, l_refiner_loss, l_terms, refiner_terms, l_image_ids

            # Unlabeled: EMA Teacher refines; Student stays on z_core and never
            # executes its refiner.
            with self._autocast():
                u_bundle = self.core_model.extract_feature_bundle(u_imgs)
                u_rgb = self.core_model.prepare_rgb(u_imgs)
            with torch.inference_mode():
                with self._autocast():
                    teacher_payload = self.core_model.teacher_pseudo(
                        u_bundle,
                        self.memory,
                        self.pc_cfg.final_epoch + epoch,
                        image_rgb=u_rgb,
                    )
            pseudo = prepare_encoder_pc_pseudo_targets(teacher_payload, self.pc_cfg)
            del teacher_payload
            with self._autocast():
                u_outputs, u_aux = self.model(
                    branch="student_unlabeled",
                    features=u_bundle,
                    image_rgb=u_rgb,
                    memory=self.memory,
                    epoch=self.pc_cfg.final_epoch + epoch,
                )
                u_loss, u_terms = encoder_pc_unlabeled_loss(
                    u_outputs,
                    u_aux,
                    pseudo,
                    self.pc_cfg,
                    epoch,
                )
            self.scaler.scale(u_loss).backward()

            # One step after both backwards, then exact-name three-module EMA.
            self.scaler.unscale_(self.optimizer)
            trainable = [
                parameter
                for group in self.optimizer.param_groups
                for parameter in group["params"]
                if parameter.requires_grad
            ]
            grad_norm = torch.nn.utils.clip_grad_norm_(
                trainable,
                float(getattr(self.cfg, "grad_clip_norm", 5.0)),
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.core_model.update_teacher(
                momentum=float(getattr(self.pc_cfg, "ema_momentum", 0.995))
            )

            u_loss_value = float(u_loss.detach())
            totals["loss"] += l_loss_value + u_loss_value
            totals["labeled"] += l_loss_value
            totals["unlabeled"] += u_loss_value
            totals["grad_norm"] += float(torch.as_tensor(grad_norm).detach())
            for name, value in u_terms.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    totals[name] += float(value.detach())
            steps += 1
            progress.set_postfix(
                loss=f"{l_loss_value + u_loss_value:.4f}",
                conf=f"{float(u_terms['pseudo_conf_mean']):.3e}",
            )
            del u_bundle, u_rgb, u_outputs, u_aux, pseudo, u_loss, u_terms, u_imgs

        if steps == 0:
            raise RuntimeError("encoder_pc TS epoch completed without optimizer steps")
        self.scheduler.step()
        self.current_epoch = epoch + 1
        means = {
            name: reduce_mean(value / steps, self.device)
            for name, value in totals.items()
        }
        return means

    def train(self) -> None:
        final_epoch = int(getattr(self.cfg, "epochs", 15))
        for epoch in range(self.current_epoch, final_epoch + 1):
            metrics = self.train_epoch(epoch)
            interval = int(getattr(self.cfg, "checkpoint_interval", 1))
            should_save = interval > 0 and epoch % interval == 0
            rank_rng = self._collect_rng_state_by_rank() if should_save else None
            if should_save and is_main_process():
                self._save_resume(epoch, metrics, rng_state_by_rank=rank_rng)
            synchronize()
        # Every rank participates in the deterministic final rebuild so the
        # internal barrier cannot deadlock.  Rank zero alone writes artifacts.
        self._rebuild_memory(
            self.core_model.student_encoder_pc_hbm,
            producer_source="student_final",
        )
        if is_main_process():
            self._finalize_artifacts(final_epoch, rebuild_memory=False)
        synchronize()

    def _artifact_meta(
        self, *, producer_fingerprint: str | None = None
    ) -> dict[str, Any]:
        base_meta = self.core_model.encoder_base_artifact_meta
        metadata = {
            "split_fingerprint": self.split_state["fingerprint"],
            "dino_weight_fingerprint": module_fingerprint(
                self.core_model.dino
            ),
            "baseline_fingerprint": str(
                base_meta.get("baseline_fingerprint", "unspecified")
            ),
            "initialization_source": "encoder_pc_base_v3",
        }
        if producer_fingerprint is not None:
            metadata["producer_fingerprint"] = str(producer_fingerprint)
        return metadata

    def _save_resume(
        self,
        epoch: int,
        metrics: Mapping[str, float],
        *,
        rng_state_by_rank: Sequence[Mapping[str, Any]] | None = None,
    ) -> Path:
        path = self.save_dir / f"encoder_pc_ts_resume_epoch_{epoch:03d}.pth"
        save_encoder_pc_training_resume(
            path,
            epoch=epoch,
            encoder_pc_hbm=self.core_model.student_encoder_pc_hbm,
            decoder=self.core_model.student,
            pseudo_refiner=self.core_model.student_pseudo_refiner,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            ema_adapter=self.core_model.teacher_encoder_pc_hbm,
            ema_decoder=self.core_model.teacher,
            ema_refiner=self.core_model.teacher_pseudo_refiner,
            config=self.pc_cfg,
            stage_state={"name": "full", "ts_epoch": int(epoch)},
            split_state=self.split_state,
            memory_profile=self.memory_profile,
            model_role="student",
            training_design=self.training_design,
            artifact_meta=self._artifact_meta(),
            extra={"metrics": dict(metrics)},
            rng_state_by_rank=rng_state_by_rank,
        )
        return path

    def resume(self, path: str | Path, *, restore_rng: bool = True) -> dict[str, Any]:
        checkpoint = load_encoder_pc_training_resume(
            path,
            encoder_pc_hbm=self.core_model.student_encoder_pc_hbm,
            decoder=self.core_model.student,
            pseudo_refiner=self.core_model.student_pseudo_refiner,
            expected_model_role="student",
            expected_training_design=self.training_design,
            expected_config=self.pc_cfg,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            ema_adapter=self.core_model.teacher_encoder_pc_hbm,
            ema_decoder=self.core_model.teacher,
            ema_refiner=self.core_model.teacher_pseudo_refiner,
            restore_rng=restore_rng,
            expected_split_state=self.split_state,
            expected_memory_profile=self.memory_profile,
            expected_artifact_meta=self._artifact_meta(),
        )
        stage_state = checkpoint.get("stage_state")
        if not isinstance(stage_state, Mapping):
            raise RuntimeError("encoder_pc TS resume is missing full stage state")
        if stage_state.get("name") != "full":
            raise RuntimeError("encoder_pc TS resume must use the full Adapter stage")
        if int(stage_state.get("ts_epoch", -1)) != int(checkpoint["epoch"]):
            raise RuntimeError("encoder_pc TS resume stage epoch is inconsistent")
        self.current_epoch = int(checkpoint["epoch"]) + 1
        self._freeze_teacher()
        return checkpoint

    @staticmethod
    def _collect_rng_state_by_rank() -> list[Mapping[str, Any]]:
        local = capture_rng_state()
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return [local]
        states: list[Mapping[str, Any] | None] = [
            None for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather_object(states, local)
        if any(state is None for state in states):
            raise RuntimeError("Failed to collect RNG state from every DDP rank")
        return [state for state in states if state is not None]

    @torch.inference_mode()
    def _finalize_artifacts(
        self, epoch: int, *, rebuild_memory: bool = True
    ) -> tuple[Path, Path]:
        if rebuild_memory:
            self._rebuild_memory(
                self.core_model.student_encoder_pc_hbm,
                producer_source="student_final",
            )
        producer_fingerprint = module_fingerprint(
            self.core_model.student_encoder_pc_hbm
        )
        if self.memory.compat_meta.get("producer_fingerprint") != producer_fingerprint:
            raise RuntimeError("Final Student artifact and memory producers differ")
        artifact_meta = self._artifact_meta(
            producer_fingerprint=producer_fingerprint
        )
        model_path = self.save_dir / "encoder_pc_ts_student_v3.pth"
        memory_path = self.save_dir / "encoder_pc_ts_memory_v3.pth"
        save_encoder_pc_checkpoint(
            model_path,
            epoch=epoch,
            encoder_pc_hbm=self.core_model.student_encoder_pc_hbm,
            decoder=self.core_model.student,
            pseudo_refiner=self.core_model.student_pseudo_refiner,
            config=self.pc_cfg,
            model_role="student",
            training_design=self.training_design,
            artifact_meta=artifact_meta,
        )
        temporary = memory_path.with_name(memory_path.name + ".tmp")
        torch.save(self.memory.state_dict(), temporary)
        temporary.replace(memory_path)
        return model_path, memory_path


Trainer = EncoderPCTSTrainer


__all__ = [
    "ENCODER_PC_TS_TRAINING_DESIGN",
    "EncoderPCTSTrainer",
    "Trainer",
]
