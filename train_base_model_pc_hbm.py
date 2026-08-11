"""Train the Base Decoder with PC-HBM-Lite."""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from configs.base_model_config import Config
from configs.pc_hbm_dino_config import DinoPCHBMConfig
from Model.base_model import BaseModel
from utils.checkpoint_pc_hbm import (
    extract_non_pc_decoder_state,
    load_decoder_compatible,
    read_pc_config,
    state_dict_fingerprint,
    validate_labeled_split_source,
)
from utils.distributed import (
    cleanup_distributed,
    configure_distributed,
    init_distributed,
    wrap_distributed,
)
from utils.trainer_base_model_pc_hbm import (
    BasePCHBMTrainer,
    configure_teacher_only_trainability,
    configure_two_stage_trainability,
)


def set_seed(seed: int = 2025, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RSBL with PC-HBM-Lite"
    )
    parser.add_argument(
        "--training-design",
        choices=("two_stage", "teacher_only"),
        default="two_stage",
    )
    parser.add_argument(
        "--output-dir",
        "--base-model-path",
        dest="output_dir",
        default="./results/base_pc_hbm_lite",
    )
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        "--decoder-checkpoint",
        default=None,
        help="Optional complete V2 PC-HBM-Lite Decoder initialization.",
    )
    initialization.add_argument(
        "--resume",
        default=None,
        help="V2 resume checkpoint produced by this entry point.",
    )
    parser.add_argument(
        "--baseline-checkpoint",
        default=None,
        help=(
            "Explicitly extract only non-pc_hbm.* Decoder weights; required "
            "for teacher_only."
        ),
    )
    parser.add_argument(
        "--init-pcv-from-legacy",
        action="store_true",
        help=(
            "Initialize a fresh parent-conditioned verifier from a complete "
            "legacy weighted_sum Decoder; retain all non-verifier weights."
        ),
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--labeled-indices-pt",
        default=None,
        help=(
            "Optional stable-key PT file. When omitted, "
            "Dataset/COD/sampled_images.txt is used."
        ),
    )
    parser.add_argument("--epochs", type=_positive_int, default=30)
    parser.add_argument("--batch-size", type=_positive_int, default=None)
    parser.add_argument(
        "--memory-batch-size", type=_positive_int, default=16
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--checkpoint-interval", type=_positive_int, default=1
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--allow-self-match", action="store_true")
    return parser.parse_args()


def validate_training_args(args: argparse.Namespace) -> None:
    if args.init_pcv_from_legacy and not args.decoder_checkpoint:
        raise ValueError(
            "--init-pcv-from-legacy requires --decoder-checkpoint"
        )
    if args.training_design == "teacher_only" and not args.baseline_checkpoint:
        raise ValueError(
            "--baseline-checkpoint is required for teacher_only"
        )
    if args.resume and args.baseline_checkpoint:
        raise ValueError("--resume cannot be combined with baseline initialization")
    if args.decoder_checkpoint and args.baseline_checkpoint:
        raise ValueError(
            "Choose either complete Lite or baseline-only initialization"
        )
    if args.resume and args.allow_self_match:
        raise ValueError(
            "--resume cannot be combined with --allow-self-match because "
            "resume requires an exact PC-HBM config"
        )


def _load_baseline_only(model: BaseModel, path: str) -> None:
    state = extract_non_pc_decoder_state(path)
    load_decoder_compatible(
        model.decoder,
        {"decoder": state},
        require_pc_complete=False,
    )


def main() -> None:
    args = parse_args()
    validate_training_args(args)
    context = init_distributed()
    try:
        cfg = Config()
        cfg.save_dir = args.output_dir
        cfg.epochs = int(args.epochs)
        cfg.train_labeled_indices_pt = args.labeled_indices_pt
        cfg.training_design = args.training_design
        cfg.memory_batch_size = int(args.memory_batch_size)
        cfg.checkpoint_interval = int(args.checkpoint_interval)
        cfg.pc_force_no_amp = bool(args.no_amp)
        if args.batch_size is not None:
            cfg.batch_size = int(args.batch_size)
        if args.num_workers is not None:
            cfg.num_workers = int(args.num_workers)
        if args.learning_rate is not None:
            cfg.learning_rate = float(args.learning_rate)
        configure_distributed(cfg, context, args.seed)
        set_seed(cfg.seed, args.deterministic)
        split_identity = validate_labeled_split_source(
            args.labeled_indices_pt,
            getattr(cfg, "train_sample_txt", None),
        )
        cfg.labeled_split_count = split_identity.count
        cfg.labeled_split_fingerprint = split_identity.fingerprint

        if args.resume:
            pc_cfg = read_pc_config(
                args.resume,
                context="Base training resume",
            )
        elif args.decoder_checkpoint:
            pc_cfg = read_pc_config(
                args.decoder_checkpoint,
                context="Base Decoder checkpoint",
                init_pcv_from_legacy=args.init_pcv_from_legacy,
            )
        else:
            pc_cfg = DinoPCHBMConfig()
        cfg.train_size = int(pc_cfg.input_size)
        model = BaseModel(pc_cfg=pc_cfg)
        if args.baseline_checkpoint:
            _load_baseline_only(model, args.baseline_checkpoint)
        elif args.decoder_checkpoint:
            load_decoder_compatible(
                model.decoder,
                args.decoder_checkpoint,
                require_pc_complete=True,
                expected_pc_cfg=pc_cfg,
                init_pcv_from_legacy=args.init_pcv_from_legacy,
            )
        pc_cfg.configure_training_design(args.training_design)
        if args.allow_self_match and not args.resume:
            pc_cfg.exclude_self_match = False
        cfg.baseline_fingerprint = state_dict_fingerprint(
            {
                name: value
                for name, value in model.decoder.state_dict().items()
                if not name.startswith("pc_hbm.")
            }
        )

        if args.training_design == "teacher_only":
            configure_teacher_only_trainability(model)
        else:
            configure_two_stage_trainability(model)
        model.to(cfg.device)
        model = wrap_distributed(
            model,
            context,
            find_unused_parameters=True,
        )
        trainer = BasePCHBMTrainer(
            model,
            cfg,
            pc_cfg,
            training_design=args.training_design,
        )
        if args.resume:
            trainer.resume(args.resume)
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
