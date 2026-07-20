"""Staged labeled trainer for the isolated encoder-side PC-HBM v3 profile."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler

from configs.pc_hbm_dino_config import EncoderPCHBMConfig
from Model.PC_HBM.encoder import (
    EncoderPCMemory,
    teacher_pseudo_refiner_labeled_loss,
)
from Model.PC_HBM.training.encoder_training import (
    EncoderPCStage,
    build_encoder_pc_optimizer,
    configure_encoder_pc_stage,
    encoder_pc_labeled_loss,
    make_ema_encoder_adapter,
    update_ema_encoder_adapter,
)
from utils.dataloader import PCLabeledTrainDataset, build_labeled_memory_loader
from utils.distributed import is_main_process, reduce_mean, synchronize, unwrap_model
from utils.checkpoint_pc_hbm import (
    capture_rng_state,
    compute_labeled_split_fingerprint,
    load_encoder_pc_training_resume,
    save_encoder_pc_checkpoint,
    save_encoder_pc_training_resume,
)
from utils.pc_memory_runner import module_fingerprint, rebuild_encoder_memory


class EncoderPCHBMTrainer:
    """Train Base epochs 1--30 while keeping DINO frozen and Decoder PC-free."""

    training_design = "two_stage"

    def __init__(
        self,
        model: nn.Module,
        cfg: Any,
        pc_cfg: EncoderPCHBMConfig | None = None,
        *,
        memory: EncoderPCMemory | None = None,
        labeled_loader=None,
        memory_loader=None,
        optimizer=None,
        scheduler=None,
        scaler=None,
        ema_adapter: nn.Module | None = None,
        memory_rebuild_fn: Callable[..., Any] = rebuild_encoder_memory,
        decoder_warm_started: bool = False,
    ) -> None:
        self.model = model
        self.model_without_ddp = unwrap_model(model)
        self.cfg = cfg
        self.pc_cfg = pc_cfg or getattr(self.model_without_ddp, "encoder_pc_config", None)
        if not isinstance(self.pc_cfg, EncoderPCHBMConfig):
            raise TypeError("EncoderPCHBMTrainer requires EncoderPCHBMConfig.")
        self.device = torch.device(getattr(cfg, "device", "cpu"))
        self.adapter = getattr(self.model_without_ddp, "encoder_pc_hbm", None)
        self.decoder = getattr(self.model_without_ddp, "decoder", None)
        self.pseudo_refiner = getattr(self.model_without_ddp, "pseudo_refiner", None)
        if not isinstance(self.adapter, nn.Module) or not isinstance(self.decoder, nn.Module):
            raise AttributeError("Encoder Base model must expose adapter and decoder modules.")
        if getattr(self.decoder, "pc_hbm", None) is not None:
            raise RuntimeError("Encoder-side profile requires decoder.pc_hbm is None.")
        if any(name.startswith("pc_hbm.") for name in self.decoder.state_dict()):
            raise RuntimeError("Encoder-side Decoder state must not contain pc_hbm.* keys.")
        dino = getattr(self.model_without_ddp, "dino", None)
        if isinstance(dino, nn.Module):
            dino.requires_grad_(False).eval()

        self.optimizer = optimizer or build_encoder_pc_optimizer(
            self.adapter,
            self.decoder,
            self.pseudo_refiner,
            decoder_warm_started=bool(decoder_warm_started),
        )
        self.scheduler = scheduler or CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, int(getattr(cfg, "epochs", self.pc_cfg.final_epoch)) - 1),
            eta_min=float(getattr(cfg, "min_lr", 1.0e-6)),
        )
        self.amp_enabled = bool(
            getattr(cfg, "use_amp", True) and self.device.type == "cuda"
        )
        self.scaler = scaler or torch.amp.GradScaler(
            "cuda", enabled=self.amp_enabled
        )
        self.ema_adapter = (
            ema_adapter if ema_adapter is not None else make_ema_encoder_adapter(self.adapter)
        )
        self.ema_adapter.to(self.device).requires_grad_(False).eval()
        self.memory = (
            memory
            if memory is not None
            else EncoderPCMemory(
                memory_dim=self.pc_cfg.memory_dim,
                value_dim=self.pc_cfg.value_dim,
                geometry_dim=self.pc_cfg.geometry_dim,
                storage_dtype=self.pc_cfg.memory_storage_dtype,
            )
        )
        self.memory_rebuild_fn = memory_rebuild_fn
        self.labeled_sampler = None
        self.labeled_loader = (
            labeled_loader if labeled_loader is not None else self._build_labeled_loader()
        )
        self.memory_loader = (
            memory_loader if memory_loader is not None else self._build_memory_loader()
        )
        if len(self.labeled_loader) == 0:
            raise ValueError("Encoder PC-HBM labeled training loader is empty.")
        self.split_state = self._build_split_state()
        self.memory_profile = {
            "schema_version": self.pc_cfg.memory_schema_version,
            "source": self.pc_cfg.memory_source,
            "storage_dtype": self.pc_cfg.memory_storage_dtype,
            "device": self.pc_cfg.memory_device,
        }
        self.current_epoch = 1
        self.last_epoch_metrics: dict[str, float] = {}
        self.save_dir = Path(getattr(cfg, "save_dir", "./results/base_encoder_pc"))
        if is_main_process():
            self.save_dir.mkdir(parents=True, exist_ok=True)
        synchronize()

    def _build_split_state(self) -> dict[str, Any]:
        dataset = getattr(self.memory_loader, "dataset", None)
        sample_keys = getattr(dataset, "sample_keys", None)
        if sample_keys:
            keys = [str(item) for item in sample_keys]
        else:
            size = len(dataset) if dataset is not None else len(self.memory_loader)
            keys = [f"@encoder-loader/{index}" for index in range(size)]
        return {
            "fingerprint": compute_labeled_split_fingerprint(keys),
            "sample_count": len(keys),
        }

    def _build_labeled_loader(self):
        dataset = PCLabeledTrainDataset(
            l_image_root=self.cfg.train_imgs,
            l_gt_root=self.cfg.train_masks,
            l_txt_root=self.cfg.train_sample_txt,
            l_train_size=self.pc_cfg.input_size,
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

    def _build_memory_loader(self):
        return build_labeled_memory_loader(
            l_image_root=self.cfg.train_imgs,
            l_gt_root=self.cfg.train_masks,
            l_txt_root=self.cfg.train_sample_txt,
            l_train_size=self.pc_cfg.input_size,
            labeled_indices_pt=getattr(self.cfg, "train_labeled_indices_pt", None),
            batch_size=int(getattr(self.cfg, "memory_batch_size", 16)),
            num_workers=int(getattr(self.cfg, "num_workers", 0)),
            pin_memory=bool(getattr(self.cfg, "CUDA", self.device.type == "cuda")),
        )

    def _rebuild_epoch_memory(self, stage: EncoderPCStage) -> None:
        if stage.mode == "bootstrap":
            return
        self.memory_rebuild_fn(
            self.model_without_ddp,
            self.ema_adapter,
            self.memory_loader,
            self.memory,
            self.device,
            config=self.pc_cfg,
            use_amp=self.amp_enabled,
        )
        if not self.memory.is_ready():
            raise RuntimeError("Encoder memory rebuild did not produce ready schema v3.")

    def train_epoch(self, epoch: int | None = None) -> dict[str, float]:
        epoch = self.current_epoch if epoch is None else int(epoch)
        if epoch != self.current_epoch:
            raise ValueError(
                f"train_epoch expected epoch {self.current_epoch}, received {epoch}."
            )
        stage = EncoderPCStage.for_epoch(epoch, self.pc_cfg)
        self.model.train()
        configure_encoder_pc_stage(
            self.adapter,
            self.decoder,
            self.pseudo_refiner,
            stage,
        )
        self._rebuild_epoch_memory(stage)
        getattr(self.model_without_ddp, "dino", nn.Identity()).eval()
        self.ema_adapter.eval()
        if self.labeled_sampler is not None:
            self.labeled_sampler.set_epoch(epoch)

        running: dict[str, float] = defaultdict(float)
        batch_count = 0
        for batch in self.labeled_loader:
            images, gt, image_ids = self._unpack_labeled_batch(batch)
            images = images.to(self.device, non_blocking=self.device.type == "cuda")
            gt = gt.to(self.device, non_blocking=self.device.type == "cuda")
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=self.amp_enabled,
            ):
                result = self.model(
                    x=images,
                    memory=None if stage.mode == "bootstrap" else self.memory,
                    pc_mode=stage.mode,
                    epoch=epoch,
                    return_aux=True,
                    query_image_ids=image_ids,
                    encoder_stage=stage.adapter_flags(
                        require_same_image_positive=stage.mode != "bootstrap"
                    ),
                    run_labeled_refiner=stage.enable_refiner,
                )
                if not isinstance(result, (tuple, list)) or len(result) != 2:
                    raise RuntimeError("Encoder Base model must return (outputs, aux).")
                outputs, aux = result
                loss, terms = encoder_pc_labeled_loss(
                    outputs, aux, gt, self.pc_cfg, stage
                )
                if stage.enable_refiner:
                    refiner_output = aux.get("pseudo_refiner")
                    if not isinstance(refiner_output, Mapping):
                        raise RuntimeError(
                            "Epochs 21-30 require labeled pseudo-refiner output."
                        )
                    refiner_loss, refiner_terms = (
                        teacher_pseudo_refiner_labeled_loss(
                            refiner_output, gt, self.pc_cfg
                        )
                    )
                    loss = loss + refiner_loss
                    terms = {**terms, **refiner_terms}
            if not bool(torch.isfinite(loss.detach())):
                raise FloatingPointError(
                    f"Non-finite encoder-PC loss at epoch {epoch}."
                )
            self.scaler.scale(loss).backward()
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
            update_ema_encoder_adapter(
                self.ema_adapter,
                self.adapter,
                decay=self.pc_cfg.memory_adapter_ema_decay,
            )
            for name, value in terms.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    running[name] += float(value.detach())
            running["loss"] += float(loss.detach())
            running["grad_norm"] += float(torch.as_tensor(grad_norm).detach())
            batch_count += 1

        if batch_count == 0:
            raise RuntimeError("Encoder PC-HBM labeled loader produced no batches.")
        metrics = {
            name: reduce_mean(value / batch_count, self.device)
            for name, value in running.items()
        }
        metrics["stage_index"] = float(stage.index)
        self.last_epoch_metrics = metrics
        self.scheduler.step()
        self.current_epoch = epoch + 1
        interval = int(getattr(self.cfg, "checkpoint_interval", 1))
        should_checkpoint = (
            self.pseudo_refiner is not None
            and interval > 0
            and epoch % interval == 0
        )
        rng_state_by_rank = (
            self._collect_rng_state_by_rank() if should_checkpoint else None
        )
        if should_checkpoint and is_main_process():
            self._save_resume(epoch, stage, rng_state_by_rank=rng_state_by_rank)
        synchronize()
        return metrics

    def train(self) -> None:
        final_epoch = int(getattr(self.cfg, "epochs", self.pc_cfg.final_epoch))
        for epoch in range(self.current_epoch, final_epoch + 1):
            self.train_epoch(epoch)
        if self.pseudo_refiner is not None and is_main_process():
            self._finalize_artifacts(final_epoch)
        synchronize()

    def _save_resume(
        self,
        epoch: int,
        stage: EncoderPCStage,
        *,
        rng_state_by_rank: Sequence[Mapping[str, Any]] | None = None,
    ) -> Path:
        if self.pseudo_refiner is None:
            raise RuntimeError("Encoder-PC v3 resume requires an explicit pseudo_refiner.")
        path = self.save_dir / f"encoder_pc_resume_epoch_{epoch:03d}.pth"
        save_encoder_pc_training_resume(
            path,
            epoch=epoch,
            encoder_pc_hbm=self.adapter,
            decoder=self.decoder,
            pseudo_refiner=self.pseudo_refiner,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            ema_adapter=self.ema_adapter,
            config=self.pc_cfg,
            stage_state={"epoch": epoch, "name": stage.name, "index": stage.index},
            split_state=self.split_state,
            memory_profile=self.memory_profile,
            model_role="base",
            training_design=self.training_design,
            artifact_meta=self._artifact_meta(),
            rng_state_by_rank=rng_state_by_rank,
        )
        return path

    @staticmethod
    def _collect_rng_state_by_rank() -> list[Mapping[str, Any]]:
        local_state = capture_rng_state()
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return [local_state]
        states: list[Mapping[str, Any] | None] = [
            None for _ in range(torch.distributed.get_world_size())
        ]
        torch.distributed.all_gather_object(states, local_state)
        if any(state is None for state in states):
            raise RuntimeError("Failed to gather RNG state from every DDP rank")
        return [state for state in states if state is not None]

    def resume(self, path: str | Path, *, restore_rng: bool = True) -> dict[str, Any]:
        if self.pseudo_refiner is None:
            raise RuntimeError("Encoder-PC v3 resume requires an explicit pseudo_refiner.")
        checkpoint = load_encoder_pc_training_resume(
            path,
            encoder_pc_hbm=self.adapter,
            decoder=self.decoder,
            pseudo_refiner=self.pseudo_refiner,
            expected_model_role="base",
            expected_training_design=self.training_design,
            expected_config=self.pc_cfg,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            ema_adapter=self.ema_adapter,
            restore_rng=restore_rng,
            expected_split_state=self.split_state,
            expected_memory_profile=self.memory_profile,
            expected_artifact_meta={
                key: value
                for key, value in self._artifact_meta().items()
                if key != "producer_fingerprint"
            },
        )
        self.current_epoch = int(checkpoint["epoch"]) + 1
        return checkpoint

    @torch.inference_mode()
    def _finalize_artifacts(self, epoch: int) -> tuple[Path, Path]:
        if self.pseudo_refiner is None:
            raise RuntimeError("Final encoder-PC artifact requires pseudo_refiner.")
        self.memory_rebuild_fn(
            self.model_without_ddp,
            self.adapter,
            self.memory_loader,
            self.memory,
            self.device,
            config=self.pc_cfg,
            use_amp=self.amp_enabled,
        )
        producer_fingerprint = module_fingerprint(self.adapter)
        if self.memory.compat_meta.get("producer_fingerprint") != producer_fingerprint:
            raise RuntimeError("Final model and memory producer fingerprints differ.")
        artifact_meta = self._artifact_meta(
            producer_fingerprint=producer_fingerprint
        )
        model_path = self.save_dir / "encoder_pc_base_v3.pth"
        memory_path = self.save_dir / "encoder_pc_memory_v3.pth"
        save_encoder_pc_checkpoint(
            model_path,
            epoch=epoch,
            encoder_pc_hbm=self.adapter,
            decoder=self.decoder,
            pseudo_refiner=self.pseudo_refiner,
            config=self.pc_cfg,
            model_role="base",
            training_design=self.training_design,
            artifact_meta=artifact_meta,
        )
        temporary = memory_path.with_name(memory_path.name + ".tmp")
        torch.save(self.memory.state_dict(), temporary)
        temporary.replace(memory_path)
        return model_path, memory_path

    def _artifact_meta(
        self, *, producer_fingerprint: str | None = None
    ) -> dict[str, Any]:
        metadata = {
            "split_fingerprint": self.split_state["fingerprint"],
            "baseline_fingerprint": str(
                getattr(self.cfg, "baseline_fingerprint", "unspecified")
            ),
            "initialization_source": str(
                getattr(self.cfg, "initialization_source", "unspecified")
            ),
        }
        if producer_fingerprint is not None:
            metadata["producer_fingerprint"] = str(producer_fingerprint)
        return metadata

    @staticmethod
    def _unpack_labeled_batch(
        batch: Sequence[Any] | Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        if isinstance(batch, Mapping):
            images = batch.get("image", batch.get("images"))
            gt = batch.get("gt", batch.get("mask"))
            image_ids = batch.get("image_ids", batch.get("sample_keys"))
        elif isinstance(batch, Sequence) and len(batch) == 4:
            _, images, gt, image_ids = batch
        elif isinstance(batch, Sequence) and len(batch) == 3:
            images, gt, image_ids = batch
        else:
            raise TypeError("Unsupported encoder labeled batch structure.")
        if not torch.is_tensor(images) or not torch.is_tensor(gt):
            raise TypeError("Encoder labeled batch must contain image and GT tensors.")
        if isinstance(image_ids, str):
            image_ids = [image_ids]
        image_ids = [str(item) for item in image_ids]
        if len(image_ids) != images.size(0):
            raise ValueError("Stable image IDs must align with labeled batch size.")
        return images, gt, image_ids


__all__ = ["EncoderPCHBMTrainer"]
