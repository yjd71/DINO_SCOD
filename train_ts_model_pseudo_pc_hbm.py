"""Teacher-only semi-supervised training entry for PC-HBM-Lite."""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from configs.pc_hbm_dino_config import DinoPCHBMConfig
from configs.ts_model_config import Config
from Model.ts_model import TSModel
from utils.checkpoint_pc_hbm import (
    validate_canonical_labeled_indices_pt,
)
from utils.distributed import (
    cleanup_distributed,
    configure_distributed,
    init_distributed,
    wrap_distributed,
)
from utils.trainer_ts_model_pseudo_pc_hbm import (
    PCHBMPseudoTrainer,
    validate_teacher_enhancer_checkpoint,
)


DEFAULT_LABELED_KEYS = (
    "./data/cache/labeled_indices/pc_bacs_0202_keys.pt"
)


def set_seed(seed=2025, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the raw Student with a PC-HBM-Lite Teacher"
    )
    parser.add_argument(
        "--training-design",
        choices=("teacher_only",),
        default="teacher_only",
    )
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--student-checkpoint", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--labeled-indices-pt",
        default=DEFAULT_LABELED_KEYS,
    )
    parser.add_argument(
        "--output-dir",
        default="./results/ts_pc_hbm_lite",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def validate_training_args(args) -> None:
    if args.training_design != "teacher_only":
        raise ValueError("PC-HBM-Lite TS supports only teacher_only")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")


def main():
    args = parse_args()
    validate_training_args(args)
    context = init_distributed()
    try:
        cfg = Config()
        cfg.save_dir = args.output_dir
        cfg.epochs = int(args.epochs)
        cfg.pc_training_design = "teacher_only"
        cfg.train_labeled_indices_pt = args.labeled_indices_pt
        cfg.teacher_pc_checkpoint = args.teacher_checkpoint
        cfg.student_checkpoint = args.student_checkpoint
        # The formal protocol fixes both physical batches at 32.
        cfg.l_batch_size = 32
        cfg.u_batch_size = 32
        if args.num_workers is not None:
            cfg.num_workers = int(args.num_workers)
        if args.learning_rate is not None:
            cfg.learning_rate = float(args.learning_rate)
        configure_distributed(cfg, context, args.seed)
        set_seed(cfg.seed, args.deterministic)

        pc_cfg = DinoPCHBMConfig(use_amp=not args.no_amp)
        pc_cfg.configure_training_design("teacher_only")
        split_fingerprint = validate_canonical_labeled_indices_pt(
            args.labeled_indices_pt
        )
        teacher_metadata = validate_teacher_enhancer_checkpoint(
            args.teacher_checkpoint,
            split_fingerprint,
        )
        cfg.labeled_split_fingerprint = split_fingerprint
        cfg.baseline_fingerprint = teacher_metadata["baseline_fingerprint"]
        model = TSModel(
            teacher_pth=args.teacher_checkpoint,
            student_pth=args.student_checkpoint,
            pc_cfg=pc_cfg,
            training_design="teacher_only",
        )
        model.to(cfg.device)
        model = wrap_distributed(
            model,
            context,
            find_unused_parameters=False,
        )
        trainer = PCHBMPseudoTrainer(
            model,
            cfg,
            pc_cfg,
            resume_path=args.resume,
        )
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
