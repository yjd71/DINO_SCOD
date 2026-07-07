import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from Model.decoder import Decoder


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

        self.patch_size = 14

        # initialize the frozen DINOv2 model: DINOv2-ViT-B/14 (default)
        self.dino = torch.hub.load('./dinov2', 'dinov2_vitb14', source='local', pretrained=False)
        self.dino.load_state_dict(torch.load('./dinov2_vitb14_pretrain.pth'))
        for param in self.dino.parameters():
            param.requires_grad = False

        self.decoder = Decoder()
        
    def forward(self, x):
        x_features = self.dino.get_intermediate_layers(x=x, n=[2, 5, 8, 11], reshape=False, return_class_token=False, norm=True)
        seg_4, seg_3, seg_2, seg_1, seg_g = self.decoder(features = x_features)

        return seg_4, seg_3, seg_2, seg_1, seg_g
    
    def inference(self, x):
        x_features = self.dino.get_intermediate_layers(x=x, n=[2, 5, 8, 11], reshape=False, return_class_token=False, norm=True)
        _, _, _, seg_1, _ = self.decoder(features = x_features)

        return seg_1
    
    def save_decoder_checkpoint(self, path):
        assert path.endswith('.pth'), f'Path should end with .pth, but got: {path}'
        torch.save(self.decoder.state_dict(), path)
        print(f'Successfully save seg parameters to {path}.')

    def load_decoder_checkpoint(self, path):
        assert path.endswith('.pth'), f'Path should end with .pth, but got: {path}'
        state_dict = torch.load(path)
        self.decoder.load_state_dict(state_dict)
        print(f'Successfully load seg parameters from {path}.')
