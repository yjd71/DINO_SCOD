import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2
from tqdm import tqdm
from utils.logging_utils import current_time
from utils.metrics import EvaluationMetricsV2
import pickle as pkl
import argparse


TABLE_COLUMNS = [
    ('Dataset', 'dataset'),
    ('Smeasure', 'sm'),
    ('meanE', 'emMean'),
    ('adpE', 'emAdp'),
    ('maxE', 'emMax'),
    ('meanF', 'fmMean'),
    ('adpF', 'fmAdp'),
    ('maxF', 'fmMax'),
    ('wF', 'wfm'),
    ('MAE', 'mae'),
]


def format_metric_value(value):
    return f'{float(value):.4f}'


def format_metric_table(rows):
    headers = [header for header, _ in TABLE_COLUMNS]
    table_rows = []
    for row in rows:
        table_rows.append([
            str(row[key]) if key == 'dataset' else format_metric_value(row[key])
            for _, key in TABLE_COLUMNS
        ])

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in table_rows))
        for i in range(len(headers))
    ]
    header_line = ' | '.join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    separator = '-+-'.join('-' * width for width in widths)
    body_lines = [
        ' | '.join(row[i].ljust(widths[i]) for i in range(len(row)))
        for row in table_rows
    ]
    return '\n'.join([header_line, separator, *body_lines])


def _non_negative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError('value must be non-negative')
    return value


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError('value must be positive')
    return value


def _read_and_validate_pair(task):
    mask_name, pred_path, mask_path = task
    pred = cv2.imread(pred_path, flags=cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
    if pred is None:
        raise FileNotFoundError(f'Failed to read prediction map: {pred_path}')
    if mask is None:
        raise FileNotFoundError(f'Failed to read GT mask: {mask_path}')
    if pred.shape != mask.shape:
        raise ValueError(
            'Prediction and GT shapes do not match: '
            f'{pred.shape}: {pred_path}\n{mask.shape}: {mask_path}'
        )
    return mask_name, pred, mask


def _iter_loaded_pairs(tasks, workers, prefetch):
    if workers == 0:
        for task in tasks:
            yield _read_and_validate_pair(task)
        return

    task_iter = iter(tasks)
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='evaluate-io')
    pending = deque()
    try:
        for _ in range(prefetch):
            try:
                pending.append(executor.submit(_read_and_validate_pair, next(task_iter)))
            except StopIteration:
                break

        while pending:
            loaded_pair = pending.popleft().result()
            try:
                pending.append(executor.submit(_read_and_validate_pair, next(task_iter)))
            except StopIteration:
                pass
            # Futures are consumed in task order, so metric.step remains deterministic.
            yield loaded_pair
    finally:
        for future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)


def evaluate(pred_root, mask_root, workers=0, prefetch=None):
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 0:
        raise ValueError(f'workers must be a non-negative integer, got {workers!r}')
    if prefetch is None:
        prefetch = max(1, workers * 2)
    if not isinstance(prefetch, int) or isinstance(prefetch, bool) or prefetch <= 0:
        raise ValueError(f'prefetch must be a positive integer, got {prefetch!r}')

    if not os.path.isdir(pred_root):
        raise FileNotFoundError(f'Prediction folder not found: {pred_root}')
    if not os.path.isdir(mask_root):
        raise FileNotFoundError(f'GT folder not found: {mask_root}')

    metric = EvaluationMetricsV2()
    mask_name_list = sorted(os.listdir(pred_root))
    mask_name_list = [name for name in mask_name_list if name.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not mask_name_list:
        raise ValueError(f'No prediction maps found in: {pred_root}')

    tasks = [
        (mask_name, os.path.join(pred_root, mask_name), os.path.join(mask_root, mask_name))
        for mask_name in mask_name_list
    ]
    loaded_pairs = _iter_loaded_pairs(tasks, workers=workers, prefetch=prefetch)
    for _, pred, mask in tqdm(loaded_pairs, total=len(tasks)):
        metric.step(pred=pred, gt=mask)

    metric_dic = metric.get_results()
    # save pickle
    with open(os.path.join(pred_root, 'evaluate.pkl'), 'wb') as f:
        pkl.dump(metric_dic, f)
    return metric_dic


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate RSBL prediction maps.')
    parser.add_argument('--pred-path', default='./results/results_random_decoder1x1/base_model/predictions')
    parser.add_argument('--datasets', nargs='+', default=['CHAMELEON', 'CAMO', 'COD10K', 'NC4K'])
    parser.add_argument(
        '--workers',
        type=_non_negative_int,
        default=4,
        help='Parallel image-reading workers; use 0 for the original sequential path (default: 4).',
    )
    parser.add_argument(
        '--prefetch',
        type=_positive_int,
        default=8,
        help='Maximum number of prediction/GT pairs queued in memory (default: 8).',
    )
    return parser.parse_args()


if __name__ == '__main__':
    from configs.base_model_config import Config

    args = parse_args()
    cfg = Config()

    print(f'{current_time()} >>> Evaluating...')
    metric_rows = []
    for dataset in args.datasets:
        metric_dic = evaluate(
            os.path.join(args.pred_path, dataset),
            getattr(cfg, f'test_{dataset}_masks'),
            workers=args.workers,
            prefetch=args.prefetch,
        )
        metric_rows.append({'dataset': dataset, **metric_dic})

    table_text = format_metric_table(metric_rows)
    txt_path = os.path.join(args.pred_path, 'evaluate_results.txt')
    os.makedirs(args.pred_path, exist_ok=True)
    print(table_text)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(table_text + '\n')

    print(f'{current_time()} >>> Evaluation results saved to: {txt_path}')
