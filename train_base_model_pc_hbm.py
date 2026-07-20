"""Train the staged labeled Base DINO PC-HBM model without changing legacy entry points."""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from configs.bgfbr_experiments import (
    build_experiment_profile,
    experiment_profile_names,
)


def set_seed(seed: int = 2025, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RSBL Base DINO PC-HBM model")
    parser.add_argument(
        "--training-design",
        choices=("two_stage", "teacher_only", "joint"),
        default="two_stage",
        help=(
            "two_stage follows the selected profile's Base schedule; teacher_only "
            "freezes the non-PC decoder; joint preserves compatibility behavior."
        ),
    )
    parser.add_argument(
        "--experiment-profile",
        choices=experiment_profile_names(),
        default="bgfbr_pc",
        help="Reproducible decoder/ablation profile (default is the full BGFBR x PC-HBM run).",
    )
    parser.add_argument(
        "--output-dir",
        "--base-model-path",
        dest="output_dir",
        default=None,
        help=(
            "Output directory. Defaults to ./results/base_pc_hbm for legacy profiles "
            "and ./results/base_encoder_pc for encoder_pc."
        ),
    )
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        "--decoder-checkpoint",
        default=None,
        help="Optional raw/nested Decoder initialization for joint compatibility mode.",
    )
    initialization.add_argument(
        "--resume",
        default=None,
        help="Resume checkpoint produced by this entry point.",
    )
    initialization.add_argument(
        "--baseline-checkpoint",
        default=None,
        help=(
            "Same-architecture Decoder initialization: required for teacher_only and "
            "optional as a two_stage warm start."
        ),
    )
    initialization.add_argument(
        "--legacy-warm-start",
        default=None,
        help=(
            "Explicit two_stage-only migration from a legacy Transformer Decoder into "
            "a BGFBR target; reuses projectors but never legacy PC weights."
        ),
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--labeled-indices-pt",
        default=None,
        help=(
            "Optional stable labeled-key/index file. When omitted, use "
            "Config.train_sample_txt (by default ./Dataset/COD/sampled_images.txt)."
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=None,
        help=(
            "Training batch size per rank/process. By default, inherit Config.batch_size "
            "(currently 16); global batch size is batch_size * world_size."
        ),
    )
    parser.add_argument("--memory-batch-size", type=_positive_int, default=16)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=_positive_int, default=1)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--allow-self-match",
        action="store_true",
        help="Disable the default same-image exclusion during labeled retrieval.",
    )
    return parser.parse_args()


def validate_training_args(args: argparse.Namespace) -> None:
    """Reject incompatible initialization contracts before model construction."""

    if args.training_design == "teacher_only":
        if not args.baseline_checkpoint:
            raise ValueError(
                "--baseline-checkpoint is required for --training-design teacher_only"
            )
        if args.decoder_checkpoint:
            raise ValueError("--decoder-checkpoint is not supported by teacher_only")
    elif args.training_design == "two_stage":
        if args.decoder_checkpoint:
            raise ValueError("--decoder-checkpoint is reserved for --training-design joint")
    elif args.baseline_checkpoint:
        raise ValueError(
            "--baseline-checkpoint is only valid for --training-design teacher_only or two_stage"
        )
    if args.legacy_warm_start and args.training_design != "two_stage":
        raise ValueError("--legacy-warm-start is only valid for --training-design two_stage")
    initialization = (
        args.legacy_warm_start,
        args.baseline_checkpoint,
        args.decoder_checkpoint,
        args.resume,
    )
    if sum(value is not None for value in initialization) > 1:
        raise ValueError(
            "--legacy-warm-start conflicts with baseline/decoder/resume initialization"
        )
    if build_experiment_profile(args.experiment_profile).pc_placement == "encoder":
        if args.training_design != "two_stage":
            raise ValueError("encoder_pc uses its fixed five-stage Base curriculum.")
        if args.decoder_checkpoint or args.legacy_warm_start:
            raise ValueError(
                "encoder_pc accepts only --baseline-checkpoint for optional BGFBR warm-start."
            )
        if args.allow_self_match:
            raise ValueError("encoder_pc always excludes retrieval self-matches.")
        if args.learning_rate is not None:
            raise ValueError(
                "encoder_pc uses fixed per-module learning rates; omit --learning-rate."
            )
        if not 1 <= int(args.epochs) <= 30:
            raise ValueError("encoder_pc Base epochs must be in [1, 30].")


if __name__ == "__main__":
    args = parse_args()
    validate_training_args(args)
    from configs.base_model_config import Config
    from configs.bgfbr_experiments import (
        apply_experiment_profile,
        build_experiment_profile,
    )
    from configs.pc_hbm_dino_config import DinoPCHBMConfig, EncoderPCHBMConfig
    from Model.base_model import BaseModel
    from utils.checkpoint_pc_hbm import (
        extract_non_pc_decoder_state,
        load_decoder_compatible,
        load_bgfbr_decoder_warm_start,
        load_legacy_into_bgfbr,
        state_dict_fingerprint,
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
    from utils.trainer_base_model_encoder_pc import EncoderPCHBMTrainer

    context = init_distributed()
    try:
        set_seed(args.seed + context.rank, deterministic=args.deterministic)
        cfg = Config()
        configure_distributed(cfg, context, seed=args.seed)
        uses_encoder_placement = (
            build_experiment_profile(args.experiment_profile).pc_placement == "encoder"
        )
        cfg.save_dir = args.output_dir or (
            "./results/base_encoder_pc"
            if uses_encoder_placement
            else "./results/base_pc_hbm"
        )
        cfg.training_design = args.training_design
        cfg.train_labeled_indices_pt = args.labeled_indices_pt
        cfg.epochs = args.epochs
        if args.batch_size is not None:
            cfg.batch_size = args.batch_size
        cfg.memory_batch_size = args.memory_batch_size
        cfg.checkpoint_interval = args.checkpoint_interval
        if args.num_workers is not None:
            cfg.num_workers = args.num_workers
        if args.learning_rate is not None:
            cfg.learning_rate = args.learning_rate

        requested_profile = build_experiment_profile(args.experiment_profile)
        encoder_profile = requested_profile.pc_placement == "encoder"
        if encoder_profile:
            experiment_profile = requested_profile
            pc_cfg = EncoderPCHBMConfig()
            cfg.use_amp = not args.no_amp
        else:
            pc_cfg = DinoPCHBMConfig(
                use_amp=not args.no_amp,
                exclude_self_match=not args.allow_self_match,
            )
            pc_cfg.configure_training_design(args.training_design)
            experiment_profile = apply_experiment_profile(pc_cfg, args.experiment_profile)
        cfg.experiment_profile = experiment_profile.name
        model = BaseModel(pc_cfg=pc_cfg).to(cfg.device)
        decoder_warm_started = False
        if encoder_profile and args.baseline_checkpoint:
            cfg.baseline_fingerprint = state_dict_fingerprint(
                extract_non_pc_decoder_state(args.baseline_checkpoint)
            )
            cfg.initialization_source = "bgfbr_non_pc_warm_start"
            load_bgfbr_decoder_warm_start(model.decoder, args.baseline_checkpoint)
            decoder_warm_started = True
        elif args.training_design in {"teacher_only", "two_stage"} and args.baseline_checkpoint:
            cfg.baseline_fingerprint = state_dict_fingerprint(
                extract_non_pc_decoder_state(args.baseline_checkpoint)
            )
            load_result = load_decoder_compatible(
                model.decoder,
                args.baseline_checkpoint,
                require_pc_complete=False,
            )
            expected_missing_pc = {
                name for name in model.decoder.state_dict() if name.startswith("pc_hbm.")
            }
            if set(load_result.missing_keys) != expected_missing_pc:
                raise RuntimeError(
                    "--baseline-checkpoint must be a complete same-architecture Decoder; "
                    "only an entirely absent pc_hbm.* subtree is permitted"
                )
        elif args.legacy_warm_start:
            if pc_cfg.decoder_arch != "bgfbr_pc_v1":
                raise ValueError(
                    "--legacy-warm-start requires a BGFBR target profile; use "
                    "--baseline-checkpoint for a same-architecture legacy_off run"
                )
            cfg.baseline_fingerprint = state_dict_fingerprint(
                extract_non_pc_decoder_state(args.legacy_warm_start)
            )
            migration = load_legacy_into_bgfbr(
                model.decoder,
                args.legacy_warm_start,
                reuse_projectors=True,
                reuse_pc_core=False,
            )
            cfg.legacy_migration_report = migration["report"]
            cfg.legacy_reused_parameter_names = migration["reused_parameter_names"]
        if encoder_profile and not decoder_warm_started:
            cfg.baseline_fingerprint = state_dict_fingerprint(model.decoder.state_dict())
            cfg.initialization_source = "scratch_bgfbr"
        if encoder_profile:
            pass
        elif args.training_design == "teacher_only":
            configure_teacher_only_trainability(model)
        elif args.training_design == "two_stage":
            configure_two_stage_trainability(model)
        elif args.decoder_checkpoint:
            load_decoder_compatible(
                model.decoder,
                args.decoder_checkpoint,
                require_pc_complete=False,
            )
        try:
            model = wrap_distributed(
                model,
                context,
                find_unused_parameters=True,
            )
        except TypeError as error:
            # Allows single-process execution while older utility code is being
            # upgraded; multi-process PC training must never omit this flag.
            if context.distributed:
                raise RuntimeError(
                    "PC-HBM DDP requires wrap_distributed(..., find_unused_parameters=True)"
                ) from error
            model = wrap_distributed(model, context)

        if encoder_profile:
            trainer = EncoderPCHBMTrainer(
                model=model,
                cfg=cfg,
                pc_cfg=pc_cfg,
                decoder_warm_started=decoder_warm_started,
            )
        else:
            trainer = BasePCHBMTrainer(
                model=model,
                cfg=cfg,
                pc_cfg=pc_cfg,
                training_design=args.training_design,
            )
        if args.resume:
            trainer.resume(args.resume)
        trainer.train()
    finally:
        cleanup_distributed()
