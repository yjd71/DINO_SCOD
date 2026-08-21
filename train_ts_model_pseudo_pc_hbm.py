"""Train Student-owned PC-HBM with an EMA Teacher and labeled-only Memory."""

import argparse
import random

import numpy as np
import torch

from configs.ts_model_config import Config
from Model.ts_model import TSModel
from utils.checkpoint_pc_hbm import (
    read_artifact_metadata,
    read_pc_config,
    validate_base_student_checkpoint,
    validate_labeled_split_source,
)
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Student-owned PC-HBM with an EMA Teacher"
    )
    initialization = parser.add_mutually_exclusive_group(required=True)
    initialization.add_argument("--base-student-checkpoint", default=None)
    initialization.add_argument("--resume", default=None)
    parser.add_argument(
        "--allow-legacy-teacher-enhancer-init",
        action="store_true",
        help=(
            "Explicitly accept a complete legacy teacher_enhancer artifact as "
            "the Base Student initialization."
        ),
    )
    parser.add_argument(
        "--labeled-indices-pt",
        default=None,
        help=(
            "Optional stable-key PT file. When omitted, "
            "Dataset/COD/sampled_images.txt is used."
        ),
    )
    parser.add_argument(
        "--output-dir", default="./results/pc_hbm_student_joint/ts"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def validate_training_args(args) -> None:
    if int(args.epochs) <= 0:
        raise ValueError("--epochs must be positive")
    if args.resume and args.allow_legacy_teacher_enhancer_init:
        raise ValueError("Legacy initialization is not valid with --resume")


def main():
    args = parse_args()
    validate_training_args(args)
    context = init_distributed()
    try:
        cfg = Config()
        cfg.save_dir = args.output_dir
        cfg.epochs = int(args.epochs)
        cfg.pc_training_design = "student_joint"
        cfg.train_labeled_indices_pt = args.labeled_indices_pt
        cfg.pc_force_no_amp = bool(args.no_amp)
        cfg.l_batch_size = 32
        cfg.u_batch_size = 32
        if args.num_workers is not None:
            cfg.num_workers = int(args.num_workers)
        if args.learning_rate is not None:
            cfg.learning_rate = float(args.learning_rate)
        configure_distributed(cfg, context, args.seed)
        set_seed(cfg.seed, args.deterministic)

        initialization_source = args.resume or args.base_student_checkpoint
        pc_cfg = read_pc_config(
            initialization_source,
            context=(
                "TS training resume" if args.resume else "Base Student checkpoint"
            ),
            allow_student_joint_defaults=bool(
                args.allow_legacy_teacher_enhancer_init
            ),
        )
        cfg.l_train_size = int(pc_cfg.input_size)
        cfg.u_train_size = int(pc_cfg.input_size)
        split_identity = validate_labeled_split_source(
            args.labeled_indices_pt, getattr(cfg, "train_sample_txt", None)
        )
        cfg.labeled_split_fingerprint = split_identity.fingerprint
        cfg.labeled_split_count = split_identity.count

        if args.resume:
            resume_metadata = read_artifact_metadata(args.resume)
            if resume_metadata is None:
                raise RuntimeError("TS resume must contain artifact metadata")
            cfg.baseline_fingerprint = resume_metadata["baseline_fingerprint"]
        else:
            base_metadata = validate_base_student_checkpoint(
                args.base_student_checkpoint,
                split_identity.fingerprint,
                allow_legacy_teacher_enhancer=bool(
                    args.allow_legacy_teacher_enhancer_init
                ),
            )
            cfg.baseline_fingerprint = base_metadata["baseline_fingerprint"]

        model = TSModel(
            base_student_pth=(None if args.resume else args.base_student_checkpoint),
            pc_cfg=pc_cfg,
            training_design="student_joint",
            allow_student_joint_defaults=bool(
                args.allow_legacy_teacher_enhancer_init
            ),
        )
        model.to(cfg.device)
        model = wrap_distributed(
            model, context, find_unused_parameters=True
        )
        trainer = PCHBMPseudoTrainer(
            model, cfg, pc_cfg, resume_path=args.resume
        )
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
