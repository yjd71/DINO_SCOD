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
from Model.PC_HBM.training.losses import pc_hbm_pc_only_labeled_loss
from configs.pc_hbm_dino_config import DinoPCHBMConfig
from utils.checkpoint_pc_hbm import (
    build_artifact_metadata,
    compute_labeled_split_fingerprint,
    compute_labeled_split_fingerprint_from_indices_pt,
    load_training_resume,
    read_artifact_metadata,
    save_decoder_checkpoint,
    save_memory_checkpoint,
    save_training_resume,
    state_dict_fingerprint,
)
from utils.dataloader import PCLabeledTrainDataset
from utils.distributed import is_main_process, reduce_mean, synchronize, unwrap_model
from utils.logging_utils import current_time
from utils.pc_memory_runner import (
    build_labeled_memory_loader,
    build_memory_compat_meta,
    rebuild_memory,
)


def configure_teacher_only_trainability(model: nn.Module) -> tuple[str, ...]:
    """Freeze the baseline model and leave only ``decoder.pc_hbm`` trainable.

    This helper is intentionally called before DDP wrapping by the dedicated
    entry point.  The trainer calls it again as an idempotent safety check for
    direct/single-process construction.
    """

    base_model = unwrap_model(model)
    decoder = getattr(base_model, "decoder", None)
    pc_hbm = getattr(decoder, "pc_hbm", None)
    if not isinstance(pc_hbm, nn.Module):
        raise RuntimeError("teacher_only training requires decoder.pc_hbm to be an nn.Module")
    base_model.requires_grad_(False)
    pc_hbm.requires_grad_(True)
    trainable = tuple(
        name for name, parameter in decoder.named_parameters() if parameter.requires_grad
    )
    if not trainable or any(not name.startswith("pc_hbm.") for name in trainable):
        raise RuntimeError(
            "teacher_only trainability must contain only non-empty pc_hbm.* parameters"
        )
    return trainable


def configure_two_stage_trainability(model: nn.Module) -> tuple[str, ...]:
    """Freeze DINO/the outer model and train the complete Decoder in both stages."""

    base_model = unwrap_model(model)
    decoder = getattr(base_model, "decoder", None)
    if not isinstance(decoder, nn.Module):
        raise RuntimeError("two_stage training requires model.decoder to be an nn.Module")
    if getattr(decoder, "pc_hbm", None) is None:
        raise RuntimeError("two_stage training requires decoder.pc_hbm")
    base_model.requires_grad_(False)
    decoder.requires_grad_(True)
    dino = getattr(base_model, "dino", None)
    if isinstance(dino, nn.Module):
        dino.requires_grad_(False)
        dino.eval()
    trainable = tuple(name for name, _ in decoder.named_parameters())
    if not trainable:
        raise RuntimeError("two_stage training requires a non-empty Decoder")
    return trainable


class BasePCHBMTrainer:
    """Train the Decoder with the selected locked 1-based PC-HBM schedule."""

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
        training_design: str | None = None,
    ) -> None:
        self.model = model
        self.cfg = cfg
        self.pc_cfg = pc_cfg or getattr(cfg, "pc_hbm", None) or DinoPCHBMConfig()
        self.training_design = str(
            training_design or getattr(cfg, "training_design", "joint")
        )
        if self.training_design not in {"two_stage", "teacher_only", "joint"}:
            raise ValueError(f"Unsupported Base PC-HBM training design: {self.training_design!r}")
        if self.training_design in {"two_stage", "teacher_only"}:
            configure = getattr(self.pc_cfg, "configure_training_design", None)
            if not callable(configure):
                raise RuntimeError(
                    f"{self.training_design} training requires "
                    "pc_cfg.configure_training_design()"
                )
            configure(self.training_design)
        if self.training_design == "teacher_only":
            configure_teacher_only_trainability(model)
        elif self.training_design == "two_stage":
            configure_two_stage_trainability(model)
        self.device = torch.device(getattr(cfg, "device", "cpu"))
        self.distributed = bool(getattr(cfg, "distributed", False))
        self.model_without_ddp = unwrap_model(model)
        if not hasattr(self.model_without_ddp, "decoder"):
            raise AttributeError("Base PC-HBM model must expose a Decoder as model.decoder")
        self.decoder = self.model_without_ddp.decoder
        if getattr(self.decoder, "pc_hbm", None) is None:
            raise RuntimeError("Base PC-HBM trainer requires BaseModel(pc_cfg=DinoPCHBMConfig(...))")
        dino = getattr(self.model_without_ddp, "dino", None)
        if isinstance(dino, nn.Module):
            dino.requires_grad_(False)
            dino.eval()

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

        dataset = self.labeled_train_set or getattr(self.labeled_train_dl, "dataset", None)
        sample_keys = getattr(dataset, "sample_keys", None)
        if sample_keys:
            cfg.labeled_split_fingerprint = compute_labeled_split_fingerprint(sample_keys)
        elif self.training_design in {"teacher_only", "two_stage"} and getattr(
            cfg, "train_labeled_indices_pt", None
        ):
            cfg.labeled_split_fingerprint = compute_labeled_split_fingerprint_from_indices_pt(
                cfg.train_labeled_indices_pt
            )
        elif not getattr(cfg, "labeled_split_fingerprint", None):
            fallback_dataset = getattr(self.labeled_train_dl, "dataset", None)
            fallback_size = len(fallback_dataset) if fallback_dataset is not None else len(
                self.labeled_train_dl
            )
            cfg.labeled_split_fingerprint = compute_labeled_split_fingerprint(
                [f"@loader/{index}" for index in range(fallback_size)]
            )
        if not getattr(cfg, "baseline_fingerprint", None):
            cfg.baseline_fingerprint = state_dict_fingerprint(
                {
                    name: value
                    for name, value in self.decoder.state_dict().items()
                    if not name.startswith("pc_hbm.")
                }
            )

        self.warning_tracker = DiagnosticWarningTracker(self.pc_cfg)
        self._diagnostic_mode: str | None = None
        self.current_epoch = 1
        self.last_epoch_metrics: dict[str, float] = {}
        self.checkpoint_metadata = dict(
            getattr(cfg, "pc_checkpoint_metadata", {}) or {}
        )
        for metadata_name in ("baseline_fingerprint", "labeled_split_fingerprint"):
            metadata_value = getattr(cfg, metadata_name, None)
            if metadata_value is not None:
                self.checkpoint_metadata.setdefault(metadata_name, str(metadata_value))
        self.checkpoint_metadata.update(
            {
                "training_design": self.training_design,
                "pc_frozen": False,
            }
        )
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
        expected_by_name = {
            name: parameter
            for name, parameter in self.decoder.named_parameters()
            if parameter.requires_grad
        }
        if self.training_design == "teacher_only" and (
            not expected_by_name
            or any(not name.startswith("pc_hbm.") for name in expected_by_name)
        ):
            raise RuntimeError(
                "teacher_only Base optimizer requires only non-empty pc_hbm.* parameters"
            )
        if self.training_design == "two_stage":
            all_decoder_parameters = dict(self.decoder.named_parameters())
            if expected_by_name.keys() != all_decoder_parameters.keys():
                frozen = sorted(all_decoder_parameters.keys() - expected_by_name.keys())
                raise RuntimeError(
                    "two_stage Base optimizer requires every Decoder parameter to be "
                    f"trainable; frozen={frozen}"
                )
        expected = {id(parameter) for parameter in expected_by_name.values()}
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
        self._set_diagnostic_mode(mode)
        self._rebuild_epoch_memory(epoch)
        if mode != "off":
            self._assert_memory_ready(epoch)

        self._set_model_train_mode()
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
                loss_function = (
                    pc_hbm_pc_only_labeled_loss
                    if self.training_design == "teacher_only"
                    else pc_hbm_labeled_loss
                )
                loss, loss_metrics = loss_function(
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
                (
                    parameter
                    for parameter in self.decoder.parameters()
                    if parameter.requires_grad
                ),
                float(self.pc_cfg.grad_clip_norm),
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._update_memory_decoder()

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
                    f"{current_time()} [Base PC-HBM] epoch={epoch} mode={mode} "
                    f"iteration={iteration} "
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
        self._update_warning_tracker(mode, epoch_metrics, emit=is_main_process())
        used_lr = float(self.optimizer.param_groups[0]["lr"])
        if self.scheduler is not None:
            self.scheduler.step()
        self.current_epoch = epoch + 1
        self._save_epoch(epoch, epoch_metrics)
        if is_main_process():
            print(
                f"{current_time()} [Base PC-HBM Epoch] epoch={epoch} mode={mode} "
                f"lr={used_lr:.8g} "
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
            print(f"{current_time()} <<< Start Base PC-HBM training; labeled={sample_count}")
        while self.current_epoch <= int(self.cfg.epochs):
            epoch = self.current_epoch
            if self.labeled_sampler is not None:
                self.labeled_sampler.set_epoch(epoch - 1)
            if is_main_process():
                print(f"{current_time()} >>> Epoch {epoch}/{self.cfg.epochs}")
            self.train_epoch(epoch)
        if self.training_design in {"teacher_only", "two_stage"}:
            self._finalize_teacher_enhancer()
        if is_main_process():
            print(f"{current_time()} <<< Base PC-HBM training finished")

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
        self._validate_resume_design(checkpoint)
        completed_epoch = int(checkpoint.get("epoch", 0))
        if completed_epoch < 0 or completed_epoch > int(self.cfg.epochs):
            raise RuntimeError(f"Invalid resume epoch: {completed_epoch}")
        self.current_epoch = completed_epoch + 1
        self.warning_tracker.history.clear()
        history = (checkpoint.get("extra") or {}).get("diagnostic_history", {})
        if isinstance(history, Mapping):
            for name, values in history.items():
                if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                    self.warning_tracker.history[str(name)].extend(float(value) for value in values)
        completed_mode = pc_mode_for_epoch(completed_epoch, self.pc_cfg)
        self._diagnostic_mode = completed_mode
        if completed_mode != "full":
            self.warning_tracker.history.clear()
        return checkpoint

    def _set_diagnostic_mode(self, mode: str) -> None:
        """Track schedule transitions and start full-mode diagnostics from a clean window."""

        mode = str(mode)
        if mode not in {"off", "parent_only", "full"}:
            raise ValueError(f"Unsupported PC-HBM diagnostic mode: {mode!r}")
        if mode == "full" and self._diagnostic_mode != "full":
            self.warning_tracker.history.clear()
        self._diagnostic_mode = mode

    def _update_warning_tracker(
        self,
        mode: str,
        metrics: Mapping[str, Any],
        *,
        emit: bool,
    ) -> list[str]:
        """Update collapse warnings only when every PC-HBM branch is active."""

        if mode != "full":
            return []
        return self.warning_tracker.update(metrics, emit=emit)

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
            artifact_meta=self._artifact_metadata("resume"),
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
            artifact_meta=self._artifact_metadata("teacher_enhancer"),
        )
        if hasattr(self.memory, "is_ready") and bool(self.memory.is_ready()):
            save_memory_checkpoint(
                self.save_dir / f"base_pc_hbm_memory_epoch_{epoch}.pth",
                self.memory,
                artifact_meta=self._artifact_metadata("teacher_memory"),
            )

    def _set_model_train_mode(self) -> None:
        self.model.train()
        dino = getattr(self.model_without_ddp, "dino", None)
        if isinstance(dino, nn.Module):
            dino.eval()
        if self.training_design == "teacher_only":
            # Frozen baseline buffers and stochastic layers must remain bitwise stable.
            self.decoder.eval()
            self.decoder.pc_hbm.train()

    @torch.no_grad()
    def _update_memory_decoder(self) -> None:
        update_ema_module(
            self.decoder,
            self.memory_decoder,
            momentum=float(self.pc_cfg.ema_momentum),
        )
        if self.training_design != "teacher_only":
            return
        # The producer tracks only the learned enhancer by EMA.  Frozen legacy
        # weights are copied exactly so repeated EMA arithmetic cannot introduce
        # rounding drift relative to the baseline checkpoint.
        source = dict(self.decoder.named_parameters())
        target = dict(self.memory_decoder.named_parameters())
        for name, source_value in source.items():
            if not name.startswith("pc_hbm."):
                target[name].copy_(source_value)

    def _finalize_teacher_enhancer(self) -> None:
        """Rebuild final memory from the main Decoder and export matched artifacts."""

        final_epoch = max(0, self.current_epoch - 1)
        compat_meta = build_memory_compat_meta(
            self.pc_cfg,
            self.decoder,
            producer_source="decoder",
        )
        self.memory_rebuild_fn(
            model=self.model,
            memory_decoder=self.decoder,
            memory_loader=self.memory_loader,
            memory=self.memory,
            device=self.device,
            config=self.pc_cfg,
            compat_meta=compat_meta,
            use_amp=self.amp_enabled,
        )
        self._assert_memory_ready(final_epoch)
        synchronize()
        if not is_main_process():
            return
        save_decoder_checkpoint(
            self.save_dir / "teacher_enhancer.pth",
            self.decoder,
            self.pc_cfg,
            final_epoch,
            artifact_meta=self._artifact_metadata("teacher_enhancer"),
        )
        save_memory_checkpoint(
            self.save_dir / "teacher_enhancer_memory.pth",
            self.memory,
            compat_meta=compat_meta,
            artifact_meta=self._artifact_metadata("teacher_memory"),
        )

    def _artifact_metadata(self, artifact_role: str) -> dict[str, Any]:
        return build_artifact_metadata(
            training_design=self.training_design,
            artifact_role=str(artifact_role),
            labeled_split_fingerprint=str(
                self.checkpoint_metadata["labeled_split_fingerprint"]
            ),
            baseline_fingerprint=str(self.checkpoint_metadata["baseline_fingerprint"]),
            pc_frozen=artifact_role in {"teacher_enhancer", "teacher_memory"},
        )

    def _validate_resume_design(self, checkpoint: Mapping[str, Any]) -> None:
        current_design = str(getattr(self, "training_design", "joint"))
        metadata = read_artifact_metadata(checkpoint)
        if metadata is None:
            if current_design != "joint":
                raise RuntimeError(
                    "Legacy resume checkpoint has no training_design; it is allowed only "
                    "with --training-design joint"
                )
            return
        saved_design = metadata["training_design"]
        if str(saved_design) != current_design:
            raise RuntimeError(
                "Cannot resume across PC-HBM training designs: "
                f"checkpoint={saved_design!r}, requested={current_design!r}"
            )
        if metadata["artifact_role"] != "resume":
            raise RuntimeError(
                f"Base resume requires artifact_role='resume', got {metadata['artifact_role']!r}"
            )
        if bool(metadata["pc_frozen"]):
            raise RuntimeError("Base resume artifact cannot mark PC-HBM as frozen")
        for key in ("labeled_split_fingerprint", "baseline_fingerprint"):
            expected = str(self.checkpoint_metadata[key])
            if str(metadata[key]) != expected:
                raise RuntimeError(
                    f"Resume {key} mismatch: checkpoint={metadata[key]!r}, expected={expected!r}"
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


__all__ = [
    "BasePCHBMTrainer",
    "Trainer",
    "configure_teacher_only_trainability",
    "configure_two_stage_trainability",
    "current_time",
]
