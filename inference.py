import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from utils.dataloader import TestDataset
from tqdm import tqdm
import os
import cv2
import argparse
import json
import time
import warnings
from collections.abc import Mapping
from contextlib import nullcontext

from configs.pc_hbm_experiments import (
    apply_experiment_profile,
    build_experiment_profile,
    experiment_profile_names,
)
from configs.pc_hbm_dino_config import DinoPCHBMConfig, EncoderPCHBMConfig
from Model.PC_HBM.encoder.encoder_memory import (
    EncoderPCMemory,
    build_encoder_memory_compat_meta,
)
from Model.PC_HBM.memory.pc_memory import PCMemory
from utils.checkpoint_pc_hbm import (
    load_decoder_compatible,
    load_encoder_pc_checkpoint,
    load_memory_checkpoint,
)
from utils.logging_utils import current_time
from utils.pc_memory_runner import module_fingerprint


LEGACY_DEFAULT_DECODER_CHECKPOINT = (
    './results/results_random_decoder1x1/ts_model_pseudo/student_epoch_15.pth'
)
ENCODER_PC_INFERENCE_CONTRACTS = {
    'base': 'two_stage',
    'student': 'teacher_student',
}
# Kept as an import-compatible alias for callers that refer specifically to
# the final Teacher/Student artifact contract.
ENCODER_PC_INFERENCE_TRAINING_DESIGN = ENCODER_PC_INFERENCE_CONTRACTS['student']
BENCHMARK_WARMUP_ITERATIONS = 10
BENCHMARK_TIMED_ITERATIONS = 50


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('value must be a positive integer')
    return value


def _non_negative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('value must be a non-negative integer')
    return value


def _inference_collate(samples):
    """Stack resized inputs while preserving variable-size output metadata."""

    ori_gts = [sample[1] for sample in samples]
    names = [sample[2] for sample in samples]
    try:
        images = torch.stack([sample[3] for sample in samples], dim=0)
    except RuntimeError as error:
        raise ValueError(
            'Inference inputs must share the configured test_size so they can be batched.'
        ) from error
    return ori_gts, names, images


def _autocast_context(device, amp):
    if bool(amp) and torch.device(device).type == 'cuda':
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    return nullcontext()


def _call_model_inference(
    model,
    images,
    *,
    memory,
    epoch,
    diagnostic_identity_fallback=False,
):
    kwargs = {'memory': memory, 'epoch': epoch}
    if diagnostic_identity_fallback:
        kwargs['allow_memory_fallback'] = True
    return model.inference(images, **kwargs)


def encoder_memory_bank_bytes(memory):
    """Return bytes owned by tensorized route/parent/child banks."""

    if memory is None:
        return 0
    total = 0
    seen = set()
    for group_name in ('route', 'parent', 'child'):
        group = getattr(memory, group_name, None)
        if not isinstance(group, Mapping):
            continue
        for value in group.values():
            if not torch.is_tensor(value):
                continue
            storage_key = (
                value.device.type,
                value.device.index,
                value.untyped_storage().data_ptr(),
                value.untyped_storage().nbytes(),
            )
            if storage_key in seen:
                continue
            seen.add(storage_key)
            total += int(value.untyped_storage().nbytes())
    return total


def benchmark_model_inference(
    model,
    images,
    *,
    memory,
    epoch=30,
    amp=False,
    diagnostic_identity_fallback=False,
):
    """Benchmark the exact formal inference call with fixed 10/50 protocol."""

    if not torch.is_tensor(images) or images.ndim != 4 or images.shape[0] <= 0:
        raise ValueError('benchmark images must be a non-empty [B,C,H,W] tensor')
    device = images.device
    cuda_device = device.type == 'cuda'
    adapter_calls = 0
    refiner_calls = 0

    def count_adapter(_module, _inputs, _output):
        nonlocal adapter_calls
        adapter_calls += 1

    def count_refiner(_module, _inputs, _output):
        nonlocal refiner_calls
        refiner_calls += 1

    hooks = []
    adapter = getattr(model, 'encoder_pc_hbm', None)
    refiner = getattr(model, 'pseudo_refiner', None)
    if isinstance(adapter, torch.nn.Module):
        hooks.append(adapter.register_forward_hook(count_adapter))
    if isinstance(refiner, torch.nn.Module):
        hooks.append(refiner.register_forward_hook(count_refiner))

    durations_ms = []
    last_output = None
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(BENCHMARK_WARMUP_ITERATIONS):
                with _autocast_context(device, amp):
                    last_output = _call_model_inference(
                        model,
                        images,
                        memory=memory,
                        epoch=epoch,
                        diagnostic_identity_fallback=diagnostic_identity_fallback,
                    )
            if cuda_device:
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            for _ in range(BENCHMARK_TIMED_ITERATIONS):
                if cuda_device:
                    torch.cuda.synchronize(device)
                started = time.perf_counter()
                with _autocast_context(device, amp):
                    last_output = _call_model_inference(
                        model,
                        images,
                        memory=memory,
                        epoch=epoch,
                        diagnostic_identity_fallback=diagnostic_identity_fallback,
                    )
                if cuda_device:
                    torch.cuda.synchronize(device)
                durations_ms.append((time.perf_counter() - started) * 1000.0)
    finally:
        for hook in hooks:
            hook.remove()

    if not torch.is_tensor(last_output) or last_output.shape[0] != images.shape[0]:
        raise RuntimeError('benchmark inference must return batched logits')
    if refiner_calls:
        raise RuntimeError(
            'Formal encoder-PC inference executed the training-only pseudo refiner.'
        )
    timings = np.asarray(durations_ms, dtype=np.float64)
    mean_ms = float(timings.mean())
    return {
        'warmup_iterations': BENCHMARK_WARMUP_ITERATIONS,
        'timed_iterations': BENCHMARK_TIMED_ITERATIONS,
        'batch_size': int(images.shape[0]),
        'mean_ms': mean_ms,
        'p50_ms': float(np.percentile(timings, 50)),
        'p95_ms': float(np.percentile(timings, 95)),
        'throughput_samples_per_s': float(images.shape[0]) * 1000.0 / mean_ms,
        'peak_cuda_memory_bytes': (
            int(torch.cuda.max_memory_allocated(device)) if cuda_device else 0
        ),
        'bank_bytes': encoder_memory_bank_bytes(memory),
        'adapter_executed': adapter_calls > 0,
        'refiner_executed': refiner_calls > 0,
    }


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
    diagnostic_identity_fallback=False,
    benchmark=False,
):
    """Run batched inference and restore every prediction to its original size.

    The programmatic defaults retain the legacy one-image behavior. The CLI
    passes throughput-oriented defaults (batch 16 and four loader workers).
    """

    if batch_size <= 0:
        raise ValueError('batch_size must be a positive integer')
    if num_workers < 0:
        raise ValueError('num_workers must be a non-negative integer')
    device = torch.device(cfg.device)
    cuda_device = device.type == 'cuda'
    amp_enabled = bool(amp and cuda_device)
    benchmark_report = None
    model.eval()
    with torch.inference_mode():
        for dataset in datasets:
            assert dataset in ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']
            save_path = os.path.join(pred_root, dataset)
            os.makedirs(save_path, exist_ok=True)

            test_dataset = TestDataset(
                image_root=getattr(cfg, f'test_{dataset}_imgs'),
                gt_root=getattr(cfg, f'test_{dataset}_masks'),
                test_size=cfg.test_size
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
                if benchmark and benchmark_report is None:
                    benchmark_report = benchmark_model_inference(
                        model,
                        images,
                        memory=memory,
                        epoch=epoch,
                        amp=amp_enabled,
                        diagnostic_identity_fallback=diagnostic_identity_fallback,
                    )
                # BaseModel.inference always returns logits. Encoder-PC returns
                # the loaded Base/Student z_core; legacy profiles retain the
                # z_main/z_final policy.
                with _autocast_context(device, amp_enabled):
                    logits = _call_model_inference(
                        model,
                        images,
                        memory=memory,
                        epoch=epoch,
                        diagnostic_identity_fallback=diagnostic_identity_fallback,
                    )
                if logits.shape[0] != len(names):
                    raise ValueError(
                        'Model inference batch dimension does not match the input batch.'
                    )
                for logit, ori_gt, name in zip(logits, ori_gts, names):
                    logit = F.interpolate(
                        logit.unsqueeze(0),
                        size=ori_gt.shape[-2:],
                        mode='bilinear',
                        align_corners=False,
                    )
                    # Keep probability conversion at this outer boundary exactly once.
                    prediction = torch.sigmoid(logit) * 255
                    prediction = prediction.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
                    output_path = os.path.join(save_path, name)
                    if not cv2.imwrite(output_path, prediction):
                        raise IOError(f'Failed to save prediction: {output_path}')
    if benchmark and benchmark_report is None:
        raise RuntimeError('Benchmark requested but no inference batch was available.')
    return benchmark_report


def parse_args():
    parser = argparse.ArgumentParser(description='Run RSBL inference.')
    parser.add_argument(
        '--experiment-profile',
        choices=experiment_profile_names(),
        default='encoder_pc',
        help='Must match the profile used to produce the Decoder and optional memory.',
    )
    parser.add_argument(
        '--model-checkpoint',
        default=None,
        help=(
            'Canonical complete Base or Student v3 adapter/decoder/refiner artifact '
            'for encoder_pc. It is invalid for decoder-side profiles.'
        ),
    )
    parser.add_argument(
        '--checkpoint',
        '--decoder-checkpoint',
        dest='decoder_checkpoint',
        default=None,
        help=(
            'Legacy decoder-side raw or nested Decoder/Student checkpoint. '
            '--checkpoint remains the legacy alias and is forbidden for encoder_pc.'
        ),
    )
    parser.add_argument(
        '--memory-checkpoint',
        default=None,
        help=(
            'Finalized PC-HBM memory checkpoint. It is strict and required for '
            'formal encoder_pc inference; legacy profiles retain warning fallback.'
        ),
    )
    parser.add_argument(
        '--epoch',
        type=int,
        default=30,
        help='Mixture schedule epoch (default: terminal Base epoch).',
    )
    parser.add_argument(
        '--require-producer-match',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Require the memory producer fingerprint to match (default: enabled).',
    )
    parser.add_argument(
        '--diagnostic-identity-fallback',
        action='store_true',
        help=(
            'Encoder-PC diagnostics only: warn and bypass an unavailable/incompatible '
            'memory as feature identity. Formal inference is strict by default.'
        ),
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run the fixed 10-warmup/50-timed model benchmark on the first batch.',
    )
    parser.add_argument(
        '--batch-size',
        type=_positive_int,
        default=16,
        help='Inference batch size (default: 16).',
    )
    parser.add_argument(
        '--num-workers',
        type=_non_negative_int,
        default=4,
        help='DataLoader worker processes (default: 4).',
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        help='Enable CUDA FP16 autocast for faster inference.',
    )
    parser.add_argument('--pred-root', default='./results/results_random_decoder1x1/ts_model_pseudo/predictions')
    parser.add_argument('--datasets', nargs='+', default=['CHAMELEON', 'CAMO', 'COD10K', 'NC4K'])
    return parser.parse_args()


def validate_inference_args(args, *, pc_enabled=None):
    """Validate profile-specific checkpoint contracts before model creation."""

    profile = build_experiment_profile(args.experiment_profile)
    prototype_disabled = pc_enabled is False
    if profile.pc_placement == 'encoder':
        if not args.model_checkpoint:
            raise ValueError('encoder_pc requires canonical --model-checkpoint.')
        if args.decoder_checkpoint:
            raise ValueError(
                '--checkpoint/--decoder-checkpoint are legacy decoder-side options '
                'and cannot be combined with encoder_pc.'
            )
        if prototype_disabled and args.memory_checkpoint:
            raise ValueError('enabled=False is a no-prototype Base run; omit --memory-checkpoint.')
        if prototype_disabled and args.diagnostic_identity_fallback:
            raise ValueError(
                'enabled=False already uses the exact Base path; diagnostic fallback is invalid.'
            )
        if (
            not prototype_disabled
            and not args.memory_checkpoint
            and not args.diagnostic_identity_fallback
        ):
            raise ValueError(
                'Formal encoder_pc inference requires --memory-checkpoint.'
            )
        if not prototype_disabled and not args.require_producer_match:
            raise ValueError(
                'encoder_pc always requires producer and split matching; '
                '--no-require-producer-match is invalid.'
            )
    else:
        if args.model_checkpoint:
            raise ValueError(
                '--model-checkpoint is reserved for encoder_pc; decoder-side '
                'profiles use --checkpoint/--decoder-checkpoint.'
            )
        if args.diagnostic_identity_fallback:
            raise ValueError(
                '--diagnostic-identity-fallback is encoder_pc-only; legacy profiles '
                'retain their existing warning fallback.'
            )
        if prototype_disabled and args.memory_checkpoint:
            raise ValueError('enabled=False is a no-prototype Base run; omit --memory-checkpoint.')
    return profile


def _load_checkpoint_mapping(source, *, artifact_name):
    if source is None:
        raise FileNotFoundError(f'{artifact_name} path was not provided.')
    if isinstance(source, Mapping):
        checkpoint = source
    else:
        checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f'{artifact_name} must contain a mapping checkpoint.')
    return checkpoint


def _required_encoder_artifact_meta(checkpoint):
    metadata = checkpoint.get('artifact_meta')
    if not isinstance(metadata, Mapping):
        raise RuntimeError('Encoder-PC model artifact is missing artifact_meta.')
    resolved = dict(metadata)
    for key in (
        'producer_fingerprint',
        'split_fingerprint',
        'dino_weight_fingerprint',
    ):
        value = resolved.get(key)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(
                f'Encoder-PC model artifact requires non-empty {key!r}.'
            )
        resolved[key] = value.strip()
    return resolved


def _encoder_pc_inference_contract(checkpoint):
    """Resolve one of the two formal encoder-PC inference artifact contracts."""

    metadata = checkpoint.get('artifact_meta')
    if not isinstance(metadata, Mapping):
        raise RuntimeError('Encoder-PC checkpoint is missing artifact_meta')
    model_role = metadata.get('model_role')
    if model_role not in ENCODER_PC_INFERENCE_CONTRACTS:
        raise RuntimeError(
            'Formal encoder-PC inference accepts model_role base or student; '
            f'got {model_role!r}.'
        )
    expected_design = ENCODER_PC_INFERENCE_CONTRACTS[model_role]
    actual_design = metadata.get('training_design')
    if actual_design != expected_design:
        raise RuntimeError(
            'Encoder-PC inference artifact role/design mismatch: '
            f'model_role={model_role!r} requires training_design='
            f'{expected_design!r}, got {actual_design!r}.'
        )
    return model_role, expected_design


def load_encoder_pc_model_for_inference(source, model, pc_cfg):
    """Strictly load a formal Base or final Student encoder-PC v3 artifact."""

    checkpoint = _load_checkpoint_mapping(
        source, artifact_name='Encoder-PC model artifact'
    )
    model_role, training_design = _encoder_pc_inference_contract(checkpoint)
    loaded = load_encoder_pc_checkpoint(
        checkpoint,
        encoder_pc_hbm=model.encoder_pc_hbm,
        decoder=model.decoder,
        pseudo_refiner=model.pseudo_refiner,
        expected_model_role=model_role,
        expected_training_design=training_design,
        expected_config=pc_cfg,
    )
    metadata = _required_encoder_artifact_meta(loaded)
    actual_producer = module_fingerprint(model.encoder_pc_hbm)
    if metadata['producer_fingerprint'] != actual_producer:
        raise RuntimeError(
            'Encoder-PC model producer fingerprint does not match the loaded '
            'Adapter state.'
        )
    actual_dino = module_fingerprint(model.dino)
    if metadata['dino_weight_fingerprint'] != actual_dino:
        raise RuntimeError(
            'Encoder-PC model DINO fingerprint does not match the live frozen DINO.'
        )
    return loaded


def load_encoder_pc_memory_for_inference(
    source,
    pc_cfg,
    *,
    model_artifact,
    encoder_pc_hbm,
    dino,
    diagnostic_identity_fallback=False,
):
    """Load schema-v3 labeled memory and cross-check it against the model."""

    try:
        metadata = _required_encoder_artifact_meta(model_artifact)
        actual_producer = module_fingerprint(encoder_pc_hbm)
        if not isinstance(dino, torch.nn.Module):
            raise TypeError('Encoder-PC inference requires the frozen DINO module.')
        if any(parameter.requires_grad for parameter in dino.parameters()):
            raise RuntimeError('Encoder-PC inference DINO weights must be frozen.')
        dino_weight_fingerprint = module_fingerprint(dino)
        if metadata['dino_weight_fingerprint'] != dino_weight_fingerprint:
            raise RuntimeError(
                'Encoder-PC model and live DINO fingerprints differ.'
            )
        if metadata['producer_fingerprint'] != actual_producer:
            raise RuntimeError(
                'Encoder-PC model producer fingerprint differs from the live Adapter.'
            )
        state = _load_checkpoint_mapping(
            source, artifact_name='Encoder-PC memory artifact'
        )
        memory = EncoderPCMemory(
            pc_cfg.memory_dim,
            pc_cfg.value_dim,
            pc_cfg.geometry_dim,
            storage_dtype=pc_cfg.memory_storage_dtype,
        )
        memory.load_state_dict(state, device='cpu', dtype=torch.float16)
        expected = build_encoder_memory_compat_meta(
            dino_weight_fingerprint=dino_weight_fingerprint,
            producer_fingerprint=actual_producer,
            labeled_split_fingerprint=metadata['split_fingerprint'],
        )
        memory.assert_compatible(
            expected,
            require_producer_match=True,
            require_split_match=True,
        )
        return memory
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        if not diagnostic_identity_fallback:
            raise
        warnings.warn(
            'Encoder-PC diagnostic identity fallback: memory is unavailable or '
            f'incompatible ({error}). Formal predictions must not use this mode.',
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def load_inference_memory(
    path,
    pc_cfg,
    require_producer_match=True,
    producer_fingerprint=None,
):
    """Load compatible CPU-FP16 memory, or warn and return ``None``.

    PC training is fail-fast, but inference remains usable through z_main when
    a memory artifact is absent, unready, or incompatible.
    """

    if path is None:
        warnings.warn(
            'No PC-HBM memory checkpoint was provided; inference will use z_main logits.',
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if require_producer_match and not producer_fingerprint:
        warnings.warn(
            'PC-HBM producer fingerprint is required for strict inference; '
            'inference will use z_main logits.',
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    memory = PCMemory(
        pc_cfg.memory_dim,
        pc_cfg.value_dim,
        pc_cfg.geometry_dim,
        storage_dtype=pc_cfg.memory_storage_dtype,
        config=pc_cfg,
    )
    try:
        load_memory_checkpoint(
            path,
            memory,
            expected_compat=pc_cfg.expected_memory_meta(
                producer_fingerprint=(
                    producer_fingerprint if require_producer_match else None
                )
            ),
            require_producer_match=require_producer_match,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        warnings.warn(
            f'PC-HBM memory is unavailable or incompatible ({error}); inference will use z_main logits.',
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return memory


if __name__ == '__main__':
    from configs.base_model_config import Config
    from Model.base_model import BaseModel
    args = parse_args()
    requested_profile = build_experiment_profile(args.experiment_profile)
    if requested_profile.pc_placement == 'encoder':
        pc_cfg = EncoderPCHBMConfig()
    else:
        pc_cfg = DinoPCHBMConfig()
    experiment_profile = validate_inference_args(
        args, pc_enabled=bool(pc_cfg.enabled)
    )
    cfg = Config()
    experiment_profile = apply_experiment_profile(pc_cfg, args.experiment_profile)
    cfg.experiment_profile = experiment_profile.name
    model = BaseModel(pc_cfg=pc_cfg)
    if not pc_cfg.enabled:
        checkpoint = (
            args.model_checkpoint
            if experiment_profile.pc_placement == 'encoder'
            else (args.decoder_checkpoint or LEGACY_DEFAULT_DECODER_CHECKPOINT)
        )
        load_decoder_compatible(
            model.decoder,
            checkpoint,
            require_pc_complete=False,
        )
        memory = None
    elif experiment_profile.pc_placement == 'encoder':
        model_artifact = load_encoder_pc_model_for_inference(
            args.model_checkpoint, model, pc_cfg
        )
        memory = load_encoder_pc_memory_for_inference(
            args.memory_checkpoint,
            pc_cfg,
            model_artifact=model_artifact,
            encoder_pc_hbm=model.encoder_pc_hbm,
            dino=model.dino,
            diagnostic_identity_fallback=args.diagnostic_identity_fallback,
        )
    else:
        decoder_checkpoint = (
            args.decoder_checkpoint or LEGACY_DEFAULT_DECODER_CHECKPOINT
        )
        load_decoder_compatible(
            model.decoder,
            decoder_checkpoint,
            require_pc_complete=args.memory_checkpoint is not None,
        )
        memory = load_inference_memory(
            args.memory_checkpoint,
            pc_cfg,
            require_producer_match=args.require_producer_match,
            producer_fingerprint=(
                module_fingerprint(model.decoder)
                if args.require_producer_match
                else None
            ),
        )
    model.to(cfg.device)
    print(f'{current_time()} >>> Inference started')
    benchmark_report = inference(
        args.datasets,
        model,
        cfg,
        args.pred_root,
        memory=memory,
        epoch=args.epoch,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        amp=args.amp,
        diagnostic_identity_fallback=args.diagnostic_identity_fallback,
        benchmark=args.benchmark,
    )
    if benchmark_report is not None:
        print(json.dumps(benchmark_report, sort_keys=True))
    print(f'{current_time()} >>> Inference finished; predictions saved to: {args.pred_root}')
