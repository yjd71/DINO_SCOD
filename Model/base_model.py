import torch
import torch.nn as nn
import warnings
from Model.BGFBR import ImageNetRGBAdapter
from Model.decoder import build_decoder, resolve_decoder_arch
from utils.checkpoint_pc_hbm import (
    load_decoder_compatible,
    save_decoder_checkpoint as save_decoder_artifact,
)


class BaseModel(nn.Module):
    def __init__(self, pc_cfg=None, decoder_arch=None, attach_pc=True):
        super(BaseModel, self).__init__()

        self.patch_size = 14

        # initialize the frozen DINOv2 model: DINOv2-ViT-B/14 (default)
        self.dino = torch.hub.load('./dinov2', 'dinov2_vitb14', source='local', pretrained=False)
        self.dino.load_state_dict(torch.load('./weight/dinov2_vitb14_pretrain.pth', map_location='cpu'))
        self.dino.requires_grad_(False)
        self.dino.eval()

        self.pc_cfg = pc_cfg
        self.decoder_arch = resolve_decoder_arch(decoder_arch, pc_cfg)
        self.rgb_adapter = ImageNetRGBAdapter()
        self.decoder = build_decoder(
            self.decoder_arch,
            pc_cfg=pc_cfg,
            attach_pc=bool(attach_pc),
        )

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()
        return self

    @torch.no_grad()
    def extract_features(self, x):
        if x.dim() != 4:
            raise ValueError(f'Expected image batch [B,C,H,W], got {tuple(x.shape)}.')
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
        return features

    def _extract_features(self, x):
        """Backward-compatible alias used by the original training scripts."""
        return self.extract_features(x)

    def prepare_rgb(self, x):
        """Return the sole RGB representation consumed by BGFBR and memory."""

        return self.rgb_adapter(x)
        
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
        image_rgb = self.prepare_rgb(x)
        return self.decoder(
            features=x_features,
            image_rgb=image_rgb,
            memory=memory,
            pc_mode=pc_mode,
            epoch=epoch,
            return_aux=return_aux,
            query_image_ids=query_image_ids,
        )
    
    def inference(self, x, memory=None, epoch=None):
        x_features = self.extract_features(x)
        image_rgb = self.prepare_rgb(x)
        if self.decoder.pc_hbm is None:
            return self.decoder(features=x_features, image_rgb=image_rgb, pc_mode='off')[3]
        if memory is None:
            warnings.warn(
                'PC-HBM memory is missing; using z_main logits.',
                RuntimeWarning,
                stacklevel=2,
            )
            return self.decoder(features=x_features, image_rgb=image_rgb, pc_mode='off')[3]
        _, aux = self.decoder(
            features=x_features,
            image_rgb=image_rgb,
            memory=memory,
            pc_mode='full',
            epoch=epoch,
            return_aux=True,
        )
        if not aux['pc_active']:
            warnings.warn(
                f'PC-HBM inference fallback: {aux.get("fallback_reason")}; '
                'using z_main logits.',
                RuntimeWarning,
                stacklevel=2,
            )
            return aux['z_main']
        return aux['z_final'] if aux['z_final'] is not None else aux['z_main']
    
    def save_decoder_checkpoint(self, path):
        assert path.endswith('.pth'), f'Path should end with .pth, but got: {path}'
        save_decoder_artifact(path, self.decoder, self.pc_cfg, epoch=0)
        print(f'Successfully save seg parameters to {path}.')

    def load_decoder_checkpoint(self, path, require_pc_complete=False):
        assert path.endswith('.pth'), f'Path should end with .pth, but got: {path}'
        load_decoder_compatible(
            self.decoder,
            path,
            require_pc_complete=bool(require_pc_complete),
        )
        print(f'Successfully load seg parameters from {path}.')
