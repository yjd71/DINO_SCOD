import argparse
import os
from contextlib import nullcontext

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.base_model_config import Config
from Model.base_model import BaseModel
from Model.PC_HBM.memory import PCMemory
from utils.checkpoint_pc_hbm import (
    load_decoder_compatible,
    load_memory_checkpoint,
    read_pc_config,
    validate_artifact_metadata,
)
from utils.dataloader import TestDataset
from utils.pc_memory_runner import module_fingerprint


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def _non_negative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return value


def _inference_collate(samples):
    """Stack resized inputs while preserving variable-size output metadata."""

    ori_gts = [sample[1] for sample in samples]
    names = [sample[2] for sample in samples]
    try:
        images = torch.stack([sample[3] for sample in samples], dim=0)
    except RuntimeError as error:
        raise ValueError(
            "Inference inputs must share the configured test_size so they can be batched."
        ) from error
    return ori_gts, names, images


def inference(
    datasets,
    model,
    cfg,
    pred_root,
    memory=None,
    epoch=30,
    batch_size=1,
    num_workers=0,
    amp=False,
    disable_pc=False,
):
    """Run strict PC inference when Memory exists, otherwise use PC off-mode."""

    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")
    effective_disable_pc = bool(disable_pc or memory is None)
    device = torch.device(cfg.device)
    cuda_device = device.type == "cuda"
    amp_enabled = bool(amp and cuda_device)
    model.eval()
    with torch.inference_mode():
        for dataset in datasets:
            if dataset not in ["CHAMELEON", "CAMO", "COD10K", "NC4K"]:
                raise ValueError(f"Unsupported COD dataset: {dataset}")
            save_path = os.path.join(pred_root, dataset)
            os.makedirs(save_path, exist_ok=True)
            test_dataset = TestDataset(
                image_root=getattr(cfg, f"test_{dataset}_imgs"),
                gt_root=getattr(cfg, f"test_{dataset}_masks"),
                test_size=cfg.test_size,
            )
            loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=cuda_device,
                persistent_workers=num_workers > 0,
                collate_fn=_inference_collate,
            )
            for ori_gts, names, images in tqdm(loader):
                images = images.to(device, non_blocking=cuda_device)
                autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if amp_enabled
                    else nullcontext()
                )
                with autocast:
                    logits = model.inference(
                        images,
                        memory=memory,
                        epoch=epoch,
                        disable_pc=effective_disable_pc,
                    )
                if logits.shape[0] != len(names):
                    raise ValueError(
                        "Model inference batch dimension does not match the input batch."
                    )
                for logit, ori_gt, name in zip(logits, ori_gts, names):
                    logit = F.interpolate(
                        logit.unsqueeze(0),
                        size=ori_gt.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                    prediction = torch.sigmoid(logit) * 255
                    prediction = (
                        prediction.squeeze(0)
                        .squeeze(0)
                        .cpu()
                        .numpy()
                        .astype(np.uint8)
                    )
                    output_path = os.path.join(save_path, name)
                    if not cv2.imwrite(output_path, prediction):
                        raise IOError(f"Failed to save prediction: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Student inference with optional PC-HBM Memory."
    )
    parser.add_argument(
        "--checkpoint",
        "--decoder-checkpoint",
        dest="decoder_checkpoint",
        required=True,
        help="Final ts_student checkpoint; --checkpoint is retained as an alias.",
    )
    parser.add_argument(
        "--memory-checkpoint",
        default=None,
        help="Matching ts_student_memory checkpoint; omit it to use PC off-mode.",
    )
    parser.add_argument(
        "--disable-pc",
        action="store_true",
        help="Explicitly force PC off-mode; omission of Memory also selects off-mode.",
    )
    parser.add_argument(
        "--allow-memory-mismatch",
        action="store_true",
        help=(
            "Explicitly allow a structurally compatible Memory produced by a "
            "different Student checkpoint. Exact producer fingerprint checks "
            "remain enabled by default."
        ),
    )
    parser.add_argument("--epoch", type=int, default=30)
    parser.add_argument("--batch-size", type=_positive_int, default=16)
    parser.add_argument("--num-workers", type=_non_negative_int, default=4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--pred-root",
        default="./results/pc_hbm_student_joint/ts/predictions",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["CHAMELEON", "CAMO", "COD10K", "NC4K"],
    )
    return parser.parse_args()


def validate_inference_args(args):
    if args.disable_pc and args.memory_checkpoint is not None:
        raise ValueError("--disable-pc cannot be combined with --memory-checkpoint")
    if (
        getattr(args, "allow_memory_mismatch", False)
        and args.memory_checkpoint is None
    ):
        raise ValueError("--allow-memory-mismatch requires --memory-checkpoint")
    return bool(args.disable_pc or args.memory_checkpoint is None)


def validate_inference_artifacts(
    decoder_path,
    memory_path=None,
    *,
    allow_memory_mismatch=False,
):
    decoder_meta = validate_artifact_metadata(
        decoder_path,
        {
            "training_design": "student_joint",
            "artifact_role": "ts_student",
            "pc_frozen": False,
        },
    )
    if memory_path is not None:
        expected_memory_meta = {
            "training_design": "student_joint",
            "artifact_role": "ts_student_memory",
            "labeled_split_fingerprint": decoder_meta[
                "labeled_split_fingerprint"
            ],
            "pc_frozen": False,
        }
        if not allow_memory_mismatch:
            expected_memory_meta["baseline_fingerprint"] = decoder_meta[
                "baseline_fingerprint"
            ]
        validate_artifact_metadata(
            memory_path,
            expected_memory_meta,
        )
    return decoder_meta


def load_inference_memory(
    path,
    pc_cfg=None,
    producer=None,
    *,
    allow_memory_mismatch=False,
):
    """Load structurally compatible memory, with strict provenance by default."""

    if path is None:
        raise ValueError("Formal inference requires a memory checkpoint")
    if pc_cfg is None:
        raise ValueError("pc_cfg is required when a memory checkpoint is supplied")
    if not isinstance(producer, torch.nn.Module):
        raise TypeError("The loaded Decoder is required to verify memory provenance")
    memory = PCMemory(config=pc_cfg)
    load_memory_checkpoint(
        path,
        memory,
        expected_compat=pc_cfg.expected_memory_meta(
            producer_fingerprint=module_fingerprint(producer)
        ),
        require_producer_match=not allow_memory_mismatch,
    )
    return memory


def main():
    args = parse_args()
    disable_pc = validate_inference_args(args)
    if args.memory_checkpoint is None and not args.disable_pc:
        print(
            "[inference] --memory-checkpoint was not provided; "
            "running the Student with PC-HBM disabled."
        )
    validate_inference_artifacts(
        args.decoder_checkpoint,
        None if disable_pc else args.memory_checkpoint,
        allow_memory_mismatch=args.allow_memory_mismatch,
    )
    if args.allow_memory_mismatch:
        print(
            "[inference] WARNING: --allow-memory-mismatch disables exact "
            "Student/Memory producer fingerprint checks. Structural PC-HBM and "
            "labeled-split compatibility are still enforced; treat the output "
            "as a mismatched-Memory ablation."
        )
    cfg = Config()
    pc_cfg = read_pc_config(
        args.decoder_checkpoint, context="TS Student inference checkpoint"
    )
    model = BaseModel(pc_cfg=pc_cfg)
    load_decoder_compatible(
        model.decoder,
        args.decoder_checkpoint,
        require_pc_complete=True,
        expected_pc_cfg=pc_cfg,
    )
    model.to(cfg.device)
    memory = (
        None
        if disable_pc
        else load_inference_memory(
            args.memory_checkpoint,
            pc_cfg=pc_cfg,
            producer=model.decoder,
            allow_memory_mismatch=args.allow_memory_mismatch,
        )
    )
    inference(
        args.datasets,
        model,
        cfg,
        args.pred_root,
        memory=memory,
        epoch=args.epoch,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=args.amp,
        disable_pc=disable_pc,
    )


if __name__ == "__main__":
    main()
