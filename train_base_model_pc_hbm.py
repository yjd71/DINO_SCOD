"""Train the staged labeled Base DINO PC-HBM model without changing legacy entry points."""

from __future__ import annotations

import argparse
import random

import numpy as np
import torch


def set_seed(seed: int = 2025, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the RSBL Base DINO PC-HBM model")
    parser.add_argument(
        "--output-dir",
        "--base-model-path",
        dest="output_dir",
        default="./results/base_pc_hbm",
        help="Directory for Decoder, memory and resumable training checkpoints.",
    )
    initialization = parser.add_mutually_exclusive_group()
    initialization.add_argument(
        "--decoder-checkpoint",
        default=None,
        help="Optional raw/nested legacy or complete PC-HBM Decoder checkpoint.",
    )
    initialization.add_argument(
        "--resume",
        default=None,
        help="Resume checkpoint produced by this entry point.",
    )
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--labeled-indices-pt", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--memory-batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--allow-self-match",
        action="store_true",
        help="Disable the default same-image exclusion during labeled retrieval.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    from configs.base_model_config import Config
    from configs.pc_hbm_dino_config import DinoPCHBMConfig
    from Model.base_model import BaseModel
    from utils.checkpoint_pc_hbm import load_decoder_compatible
    from utils.distributed import (
        cleanup_distributed,
        configure_distributed,
        init_distributed,
        wrap_distributed,
    )
    from utils.trainer_base_model_pc_hbm import BasePCHBMTrainer

    context = init_distributed()
    try:
        set_seed(args.seed + context.rank, deterministic=args.deterministic)
        cfg = Config()
        configure_distributed(cfg, context, seed=args.seed)
        cfg.save_dir = args.output_dir
        cfg.train_labeled_indices_pt = args.labeled_indices_pt
        cfg.epochs = args.epochs
        cfg.memory_batch_size = args.memory_batch_size
        cfg.checkpoint_interval = args.checkpoint_interval
        if args.num_workers is not None:
            cfg.num_workers = args.num_workers
        if args.learning_rate is not None:
            cfg.learning_rate = args.learning_rate

        pc_cfg = DinoPCHBMConfig(
            use_amp=not args.no_amp,
            exclude_self_match=not args.allow_self_match,
        )
        model = BaseModel(pc_cfg=pc_cfg).to(cfg.device)
        if args.decoder_checkpoint:
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

        trainer = BasePCHBMTrainer(model=model, cfg=cfg, pc_cfg=pc_cfg)
        if args.resume:
            trainer.resume(args.resume)
        trainer.train()
    finally:
        cleanup_distributed()
