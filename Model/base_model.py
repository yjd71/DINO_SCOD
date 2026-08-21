import torch
import torch.nn as nn
import warnings
from Model.decoder import Decoder
from utils.checkpoint_pc_hbm import load_decoder_compatible


class BaseModel(nn.Module):
    def __init__(self, pc_cfg=None):
        super(BaseModel, self).__init__()

        self.patch_size = 14

        # initialize the frozen DINOv2 model: DINOv2-ViT-B/14 (default)
        self.dino = torch.hub.load('./dinov2', 'dinov2_vitb14', source='local', pretrained=False)
        self.dino.load_state_dict(torch.load('./weight/dinov2_vitb14_pretrain.pth', map_location='cpu'))
        self.dino.requires_grad_(False)
        self.dino.eval()

        self.pc_cfg = pc_cfg
        decoder_kwargs = {}
        if pc_cfg is not None and getattr(pc_cfg, 'enabled', False):
            decoder_kwargs = {
                'in_dim': int(pc_cfg.encoder_dim),
                'out_dim': int(pc_cfg.decoder_dim),
            }
        self.decoder = Decoder(pc_cfg=pc_cfg, **decoder_kwargs)

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()
        return self

    @torch.no_grad()
    def extract_features(self, x):
        if x.dim() != 4:
            raise ValueError(f'Expected image batch [B,C,H,W], got {tuple(x.shape)}.')
        if x.size(1) != 3:
            raise ValueError(f'Expected three image channels, got {x.size(1)}.')
        if (
            self.pc_cfg is not None
            and getattr(self.pc_cfg, 'enabled', False)
            and tuple(x.shape[-2:])
            != (self.pc_cfg.input_size, self.pc_cfg.input_size)
        ):
            raise ValueError(
                f'PC-HBM-Lite requires {self.pc_cfg.input_size}x'
                f'{self.pc_cfg.input_size} input, '
                f'got {tuple(x.shape[-2:])}.'
            )
        with torch.no_grad():
            layer_indices = (
                list(self.pc_cfg.dino_layer_indices)
                if self.pc_cfg is not None
                else [2, 5, 8, 11]
            )
            features = self.dino.get_intermediate_layers(
                x=x,
                n=layer_indices,
                reshape=False,
                return_class_token=False,
                norm=True,
            )
        if len(features) != 4:
            raise RuntimeError(f'DINO returned {len(features)} feature levels instead of four.')
        if self.pc_cfg is not None and getattr(self.pc_cfg, 'enabled', False):
            expected_shape = (
                x.size(0),
                self.pc_cfg.token_size * self.pc_cfg.token_size,
                self.pc_cfg.encoder_dim,
            )
            for index, feature in enumerate(features):
                if tuple(feature.shape) != expected_shape:
                    raise RuntimeError(
                        f'DINO layer {self.pc_cfg.dino_layer_indices[index]} '
                        f'must be {expected_shape}, got {tuple(feature.shape)}.'
                    )
        return features

    def forward(
        self,
        x,
        memory=None,
        pc_mode='off',
        epoch=None,
        return_aux=False,
        query_image_ids=None,
    ):
        x_features = self.extract_features(x)
        return self.decoder(
            features=x_features,
            memory=memory,
            pc_mode=pc_mode,
            epoch=epoch,
            return_aux=return_aux,
            query_image_ids=query_image_ids,
        )
    
    def inference(self, x, memory=None, epoch=None, disable_pc=False):
        x_features = self.extract_features(x)
        if disable_pc:
            return self.decoder(features=x_features, pc_mode='off')[3]
        if self.decoder.pc_hbm is None:
            raise RuntimeError('Formal inference requires a Decoder with PC-HBM attached.')
        if memory is None:
            raise RuntimeError('Formal inference requires finalized PC-HBM memory.')
        _, aux = self.decoder(
            features=x_features,
            memory=memory,
            pc_mode='full',
            epoch=epoch,
            return_aux=True,
        )
        if not aux['pc_active'] or aux.get('pc_engine_source') != 'internal_trainable':
            raise RuntimeError('Formal inference did not execute the internal PC-HBM path.')
        return aux['z_final']
    
    def save_decoder_checkpoint(self, path):
        assert path.endswith('.pth'), f'Path should end with .pth, but got: {path}'
        torch.save(self.decoder.state_dict(), path)
        print(f'Successfully save seg parameters to {path}.')

    def load_decoder_checkpoint(self, path, require_pc_complete=False):
        assert path.endswith('.pth'), f'Path should end with .pth, but got: {path}'
        load_decoder_compatible(
            self.decoder,
            path,
            require_pc_complete=bool(require_pc_complete),
            expected_pc_cfg=(
                self.pc_cfg if bool(require_pc_complete) else None
            ),
        )
        print(f'Successfully load seg parameters from {path}.')
