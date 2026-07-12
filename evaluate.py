import os
import cv2
from tqdm import tqdm
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


def evaluate(pred_root, mask_root):
    if not os.path.isdir(pred_root):
        raise FileNotFoundError(f'Prediction folder not found: {pred_root}')
    if not os.path.isdir(mask_root):
        raise FileNotFoundError(f'GT folder not found: {mask_root}')

    metric = EvaluationMetricsV2()
    mask_name_list = sorted(os.listdir(pred_root))
    mask_name_list = [name for name in mask_name_list if name.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not mask_name_list:
        raise ValueError(f'No prediction maps found in: {pred_root}')

    for mask_name in tqdm(mask_name_list):
        pred_path = os.path.join(pred_root, mask_name)
        mask_path = os.path.join(mask_root, mask_name)
        pred = cv2.imread(pred_path, flags=cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        if pred is None:
            raise FileNotFoundError(f'Failed to read prediction map: {pred_path}')
        if mask is None:
            raise FileNotFoundError(f'Failed to read GT mask: {mask_path}')
        assert pred.shape == mask.shape, f'{pred.shape}: {pred_path}\n{mask.shape}: {mask_path}'
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
    return parser.parse_args()


if __name__ == '__main__':
    from configs.base_model_config import Config

    args = parse_args()
    cfg = Config()

    print(f'Evaluating...')
    metric_rows = []
    for dataset in args.datasets:
        metric_dic = evaluate(os.path.join(args.pred_path, dataset), getattr(cfg, f'test_{dataset}_masks'))
        metric_rows.append({'dataset': dataset, **metric_dic})

    table_text = format_metric_table(metric_rows)
    txt_path = os.path.join(args.pred_path, 'evaluate_results.txt')
    os.makedirs(args.pred_path, exist_ok=True)
    print(table_text)
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(table_text + '\n')

    print(f'Evaluation results saved to: {txt_path}')
