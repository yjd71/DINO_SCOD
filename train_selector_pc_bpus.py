from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from Model.PC_HBM.training.losses import base_structure_loss
from Model.base_model import BaseModel
from selection.artifacts import (
    atomic_json_save,
    atomic_torch_save,
    compute_catalog_fingerprint,
    file_sha256,
    load_split_keys,
)
from selection.protocol import SamplingProtocol
from utils.checkpoint_pc_hbm import (
    compute_labeled_split_fingerprint,
    state_dict_fingerprint,
)
from utils.dataloader import LabeledTrainDataset, SelectionPoolDataset


REPO_ROOT = Path(__file__).resolve().parent
DINO_WEIGHT_PATH = REPO_ROOT / "weight" / "dinov2_vitb14_pretrain.pth"
SELECTOR_SCHEMA_VERSION = 1
FORMAL_CATALOG_SIZE = 4040


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the frozen-DINO legacy Selector used by PC-BPUS"
    )
    parser.add_argument("--data-root", type=Path, default=Path("./Dataset/COD"))
    parser.add_argument(
        "--train-sets", nargs="+", default=["TR-CAMO", "TR-COD10K"]
    )
    parser.add_argument(
        "--target-counts",
        nargs=3,
        type=int,
        required=True,
        metavar=("BOOTSTRAP", "MID", "MAX"),
    )
    parser.add_argument(
        "--debug-custom-counts",
        action="store_true",
        help="Allow non-formal counts only for an isolated debug output directory.",
    )
    parser.add_argument("--labeled-indices-pt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=_positive_int, default=15)
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--learning-rate", type=_positive_float, default=1.0e-4)
    parser.add_argument("--min-learning-rate", type=_positive_float, default=1.0e-7)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument(
        "--amp", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--deterministic", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Avoid the fused/memory-efficient SDPA backward kernels, which PyTorch
        # explicitly marks as non-deterministic on CUDA.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=False)


def _worker_init(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _resolve_paths(args: argparse.Namespace) -> None:
    for name in ("data_root", "labeled_indices_pt", "output_dir", "resume"):
        value = getattr(args, name, None)
        if value is not None:
            path = Path(value)
            if not path.is_absolute():
                path = REPO_ROOT / path
            setattr(args, name, path.resolve())


def _image_roots(args: argparse.Namespace) -> list[str]:
    return [str(args.data_root / subset / "im") for subset in args.train_sets]


def _mask_roots(args: argparse.Namespace) -> list[str]:
    return [str(args.data_root / subset / "gt") for subset in args.train_sets]


def _resume_payload(
    *,
    epoch: int,
    model: BaseModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    seed: int,
    target_counts: tuple[int, ...],
    split_fingerprint: str,
    dino_fingerprint: str,
    catalog_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema_version": SELECTOR_SCHEMA_VERSION,
        "kind": "pc_bpus_selector_resume",
        "completed_epoch": int(epoch),
        "decoder_state": model.decoder.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "seed": int(seed),
        "target_counts": list(target_counts),
        "split_fingerprint": split_fingerprint,
        "dino_weight_fingerprint": dino_fingerprint,
        "catalog_fingerprint": catalog_fingerprint,
    }


def _load_resume(
    path: Path,
    *,
    model: BaseModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    seed: int,
    target_counts: tuple[int, ...],
    split_fingerprint: str,
    dino_fingerprint: str,
    catalog_fingerprint: str,
) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("kind") != "pc_bpus_selector_resume":
        raise ValueError(f"Unsupported PC-BPUS selector resume payload: {path}")
    expected = {
        "seed": int(seed),
        "target_counts": list(target_counts),
        "split_fingerprint": split_fingerprint,
        "dino_weight_fingerprint": dino_fingerprint,
        "catalog_fingerprint": catalog_fingerprint,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(
                f"Selector resume {key} mismatch: {payload.get(key)!r} != {value!r}"
            )
    model.decoder.load_state_dict(payload["decoder_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    scaler.load_state_dict(payload["scaler_state"])
    return int(payload["completed_epoch"]) + 1


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    _resolve_paths(args)
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.min_learning_rate > args.learning_rate:
        raise ValueError("--min-learning-rate must not exceed --learning-rate")
    protocol = SamplingProtocol.from_counts(
        args.target_counts, allow_custom=bool(args.debug_custom_counts)
    )
    if not protocol.is_formal and "debug" not in str(args.output_dir).lower():
        raise ValueError(
            "Custom counts require an isolated output directory containing 'debug'"
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    if not DINO_WEIGHT_PATH.is_file():
        raise FileNotFoundError(f"Missing DINOv2 weight: {DINO_WEIGHT_PATH}")
    if not args.labeled_indices_pt.is_file():
        raise FileNotFoundError(
            f"Missing bootstrap split: {args.labeled_indices_pt}"
        )

    os.chdir(REPO_ROOT)
    full_pool = SelectionPoolDataset(_image_roots(args), image_size=392)
    catalog_keys = list(full_pool.sample_keys)
    if catalog_keys != sorted(catalog_keys) or len(catalog_keys) != len(set(catalog_keys)):
        raise RuntimeError("Selector catalog must be sorted and unique")
    if protocol.is_formal and len(catalog_keys) != FORMAL_CATALOG_SIZE:
        raise RuntimeError(
            f"Formal Selector training requires {FORMAL_CATALOG_SIZE} images, "
            f"found {len(catalog_keys)}"
        )
    catalog_fingerprint = compute_catalog_fingerprint(catalog_keys)
    bootstrap_keys = load_split_keys(
        args.labeled_indices_pt, catalog_keys=catalog_keys
    )
    if len(bootstrap_keys) != protocol.bootstrap_count:
        raise ValueError(
            f"Bootstrap contains {len(bootstrap_keys)} keys; protocol requires "
            f"{protocol.bootstrap_count}."
        )
    split_fingerprint = compute_labeled_split_fingerprint(bootstrap_keys)
    dino_fingerprint = file_sha256(DINO_WEIGHT_PATH)

    labeled_dataset = LabeledTrainDataset(
        _image_roots(args),
        _mask_roots(args),
        None,
        392,
        labeled_indices_pt=str(args.labeled_indices_pt),
        rVFlip=True,
        rCrop=True,
        rRotate=False,
        colorEnhance=True,
        rPeper=False,
    )
    if len(labeled_dataset) != protocol.bootstrap_count:
        raise RuntimeError(
            f"Labeled dataset resolved {len(labeled_dataset)} samples instead of "
            f"{protocol.bootstrap_count}."
        )

    print(
        f"PC-BPUS Selector protocol={protocol.name} seed={args.seed} "
        f"catalog={len(catalog_keys)} labeled={len(labeled_dataset)} "
        f"split={split_fingerprint[:12]}"
    )
    if args.dry_run:
        print("Dry run passed; no files were written.")
        return 0

    if args.output_dir.exists() and not args.resume and not args.overwrite:
        existing = [path for path in args.output_dir.iterdir()]
        if existing:
            raise FileExistsError(
                f"Selector output directory is not empty: {args.output_dir}. "
                "Use --resume or --overwrite explicitly."
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(args.seed, bool(args.deterministic))
    device = torch.device(args.device)

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    loader = DataLoader(
        labeled_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=args.num_workers > 0,
        worker_init_fn=_worker_init,
        generator=generator,
    )

    model = BaseModel(pc_cfg=None).to(device)
    model.dino.requires_grad_(False)
    model.dino.eval()
    if getattr(model.decoder, "pc_hbm", None) is not None:
        raise RuntimeError("PC-BPUS Selector must not attach PC-HBM")
    optimizer = torch.optim.Adam(
        model.decoder.parameters(), lr=args.learning_rate, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.min_learning_rate
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = 1
    if args.resume:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Missing selector resume file: {args.resume}")
        start_epoch = _load_resume(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            seed=args.seed,
            target_counts=protocol.target_counts,
            split_fingerprint=split_fingerprint,
            dino_fingerprint=dino_fingerprint,
            catalog_fingerprint=catalog_fingerprint,
        )
    if start_epoch > args.epochs:
        raise ValueError(
            f"Resume already completed epoch {start_epoch - 1}, beyond --epochs {args.epochs}."
        )

    log_path = args.output_dir / "training_log.txt"
    log_mode = "a" if args.resume else "w"
    with log_path.open(log_mode, encoding="utf-8") as log_file:
        for epoch in range(start_epoch, args.epochs + 1):
            started = time.perf_counter()
            model.train()
            model.dino.eval()
            total_loss = 0.0
            batch_count = 0
            for _, images, gt in loader:
                images = images.to(device, non_blocking=device.type == "cuda")
                gt = gt.to(device, non_blocking=device.type == "cuda")
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    outputs = model(images, pc_mode="off", return_aux=False)
                    loss = base_structure_loss(outputs, gt)
                if not bool(torch.isfinite(loss.detach())):
                    raise FloatingPointError(
                        f"Non-finite Selector loss at epoch={epoch}, batch={batch_count + 1}"
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach().cpu())
                batch_count += 1
            if batch_count == 0:
                raise RuntimeError("Selector DataLoader produced no batches")
            used_lr = float(optimizer.param_groups[0]["lr"])
            scheduler.step()
            record = {
                "epoch": epoch,
                "loss": total_loss / batch_count,
                "learning_rate": used_lr,
                "seconds": time.perf_counter() - started,
            }
            line = json.dumps(record, ensure_ascii=False, sort_keys=True)
            print(line)
            log_file.write(line + "\n")
            log_file.flush()
            atomic_torch_save(
                _resume_payload(
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    seed=args.seed,
                    target_counts=protocol.target_counts,
                    split_fingerprint=split_fingerprint,
                    dino_fingerprint=dino_fingerprint,
                    catalog_fingerprint=catalog_fingerprint,
                ),
                args.output_dir / "selector_resume.pth",
                refuse_mismatch=False,
            )

    raw_state = {
        key: value.detach().cpu() for key, value in model.decoder.state_dict().items()
    }
    if any(key.startswith("pc_hbm.") for key in raw_state):
        raise RuntimeError("selector_raw.pth unexpectedly contains PC-HBM parameters")
    atomic_torch_save(
        raw_state,
        args.output_dir / "selector_raw.pth",
        refuse_mismatch=not args.overwrite,
    )
    selector_fingerprint = state_dict_fingerprint(raw_state)
    config_payload = {
        "schema_version": SELECTOR_SCHEMA_VERSION,
        "kind": "pc_bpus_selector",
        "protocol": protocol.name,
        "target_counts": list(protocol.target_counts),
        "bootstrap_count": protocol.bootstrap_count,
        "seed": args.seed,
        "train_sets": list(args.train_sets),
        "catalog_count": len(catalog_keys),
        "catalog_fingerprint": catalog_fingerprint,
        "split_path": str(args.labeled_indices_pt),
        "split_fingerprint": split_fingerprint,
        "dino_weight_path": str(DINO_WEIGHT_PATH),
        "dino_weight_fingerprint": dino_fingerprint,
        "selector_fingerprint": selector_fingerprint,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "optimizer": "Adam",
        "scheduler": "CosineAnnealingLR",
        "pc_mode": "off",
        "amp": amp_enabled,
        "deterministic": bool(args.deterministic),
    }
    atomic_json_save(
        config_payload,
        args.output_dir / "selector_config.json",
        refuse_mismatch=not args.overwrite,
    )
    (args.output_dir / "selector_split_fingerprint.txt").write_text(
        split_fingerprint + "\n", encoding="utf-8"
    )
    print(
        f"Saved Selector to {args.output_dir} "
        f"fingerprint={selector_fingerprint}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
