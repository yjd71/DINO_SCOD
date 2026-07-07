import os
import cv2
from tqdm import tqdm
from utils.metrics import EvaluationMetricsV2
import pickle as pkl


def evaluate(pred_root, mask_root):
    metric = EvaluationMetricsV2()
    mask_name_list = sorted(os.listdir(pred_root))
    mask_name_list = [name for name in mask_name_list if name.endswith('.png') or name.endswith('.jpg')]

    for i, mask_name in tqdm(list(enumerate(mask_name_list))):
        pred_path = os.path.join(pred_root, mask_name)
        mask_path = os.path.join(mask_root, mask_name)
        pred = cv2.imread(pred_path, flags=cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, flags=cv2.IMREAD_GRAYSCALE)
        assert pred.shape == mask.shape, f'{pred.shape}: {pred_path}\n{mask.shape}: {mask_path}'
        metric.step(pred=pred, gt=mask)

    metric_dic = metric.get_results()
    # save pickle
    with open(os.path.join(pred_root, 'evaluate.pkl'), 'wb') as f:
        pkl.dump(metric_dic, f)
    return metric_dic


if __name__ == '__main__':
    from configs.base_model_config import Config

    cfg = Config()
    pred_path = 'path to your prediction maps'

    datasets = ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']
    print(f'Evaluating...')
    for dataset in datasets:
        metric_dic = evaluate(os.path.join(pred_path, dataset), getattr(cfg, f'test_{dataset}_masks'))

        sm = metric_dic['sm']

        emMean = metric_dic['emMean']
        emAdp = metric_dic['emAdp']
        emMax = metric_dic['emMax']

        fmMean = metric_dic['fmMean']
        fmAdp = metric_dic['fmAdp']
        fmMax = metric_dic['fmMax']

        wfm = metric_dic['wfm']
        mae = metric_dic['mae']

        print(f'##### {dataset} #####')
        print(f'sm: {sm}')
        print(f'emMean: {emMean}')
        print(f'emAdp: {emAdp}')
        print(f'emMax: {emMax}')
        print(f'fmMean: {fmMean}')
        print(f'fmAdp: {fmAdp}')
        print(f'fmMax: {fmMax}')
        print(f'wfm: {wfm}')
        print(f'mae: {mae}')
