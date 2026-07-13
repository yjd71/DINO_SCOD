"""Dedicated DINO PC-HBM online-pseudo training entry point."""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from configs.ts_model_config import Config
from Model.ts_model import TSModel
from utils.distributed import (
    cleanup_distributed,
    configure_distributed,
    init_distributed,
    wrap_distributed,
)
from utils.trainer_ts_model_pseudo_pc_hbm import PCHBMPseudoTrainer


def set_seed(seed=2025, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RSBL DINO PC-HBM with online EMA-Teacher pseudo labels."
    )
    parser.add_argument(
        "--training-design",
        choices=("teacher_only", "joint"),
        default="teacher_only",
        help="Teacher-only distillation is the default; joint preserves the legacy PC-HBM flow.",
    )
    parser.add_argument(
        "--teacher-pc-checkpoint",
        "--base-pc-checkpoint",
        dest="teacher_pc_checkpoint",
        required=True,
        help="Complete Base PC-HBM Teacher/enhancer checkpoint.",
    )
    parser.add_argument(
        "--student-checkpoint",
        default=None,
        help="Optional raw Student checkpoint; defaults to non-PC weights from the Teacher checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default="./results/pc_hbm/ts_model",
        help="Directory for Student, memory, and resume checkpoints.",
    )
    parser.add_argument("--resume", default=None, help="Optional TS PC-HBM resume checkpoint.")
    parser.add_argument(
        "--allow-legacy-pc-init",
        action="store_true",
        help="Explicit migration override: initialize missing PC-HBM weights randomly.",
    )
    parser.add_argument(
        "--labeled-indices-pt",
        default=None,
        help="Optional stable labeled-key/index file; overrides sampled_images.txt selection.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--memory-batch-size", type=int, default=16)
    parser.add_argument("--memory-num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic cuDNN kernels; this is slower.",
    )
    return parser.parse_args()


def validate_training_args(args) -> None:
    if args.training_design == "teacher_only":
        if not args.labeled_indices_pt:
            raise ValueError("teacher_only TS requires --labeled-indices-pt")
        if args.allow_legacy_pc_init:
            raise ValueError("--allow-legacy-pc-init is only valid with --training-design joint")


def main():
    args = parse_args()
    validate_training_args(args)
    context = init_distributed()
    process_seed = int(args.seed) + int(context.rank)
    set_seed(process_seed, deterministic=args.deterministic)

    cfg = Config()
    cfg.epochs = int(args.epochs)
    cfg.num_workers = int(args.num_workers)
    cfg.memory_batch_size = int(args.memory_batch_size)
    cfg.memory_num_workers = int(args.memory_num_workers)
    cfg.train_labeled_indices_pt = args.labeled_indices_pt
    cfg.save_dir = args.output_dir
    cfg.pc_training_design = str(args.training_design)
    cfg.teacher_pc_checkpoint = args.teacher_pc_checkpoint
    cfg.student_checkpoint = args.student_checkpoint
    # Locked protocol: do not expose a batch-size downgrade through this CLI.
    cfg.l_batch_size = 32
    cfg.u_batch_size = 32
    configure_distributed(cfg, context, seed=int(args.seed))

    pc_cfg = DinoPCHBMConfig()
    pc_cfg.configure_training_design(args.training_design)
    model = TSModel(
        teacher_pth=args.teacher_pc_checkpoint,
        student_pth=args.student_checkpoint,
        pc_cfg=pc_cfg,
        allow_legacy_pc_init=bool(args.allow_legacy_pc_init),
        training_design=args.training_design,
    ).to(context.device)
    model = wrap_distributed(
        model,
        context,
        find_unused_parameters=args.training_design == "joint",
    )
    trainer = PCHBMPseudoTrainer(
        model,
        cfg,
        pc_cfg,
        resume_path=args.resume,
    )
    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
