import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import inference
from configs.base_model_config import Config
from Model.base_model import BaseModel
from utils.decoer_1x1cov import Conv1x1Decoder
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Run RSBL inference.')
    parser.add_argument('--checkpoint', default='./results/results_random_decoder1x1/base_model/base_model_epoch_30.pth')
    parser.add_argument('--pred-root', default='./results/results_random_decoder1x1/base_model/predictions')
    parser.add_argument('--datasets', nargs='+', default=['CHAMELEON', 'CAMO', 'COD10K', 'NC4K'])
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    cfg = Config()

    model = BaseModel()
    model.decoder = Conv1x1Decoder()
    model.load_decoder_checkpoint(args.checkpoint)
    model.to(cfg.device)

    inference(args.datasets, model, cfg, args.pred_root)
