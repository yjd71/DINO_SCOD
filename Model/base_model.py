import torch
import torch.nn as nn
import warnings
from Model.BGFBR import ImageNetRGBAdapter
from Model.decoder import build_decoder, resolve_decoder_arch
from configs.pc_hbm_dino_config import EncoderPCHBMConfig
from Model.PC_HBM.encoder import (
    DinoFeatureBundle,
    EncoderPCCoreResult,
    EncoderPCHBMAdapter,
    EncoderPCSegmentationHead,
    EncoderPCStageFlags,
    TeacherPseudoLabelRefiner,
)
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
        self.pc_placement = str(getattr(pc_cfg, 'pc_placement', 'decoder'))
        if self.pc_placement not in {'decoder', 'encoder'}:
            raise ValueError(
                f'Unsupported pc_placement={self.pc_placement!r}; expected '
                "'decoder' or 'encoder'."
            )
        self.rgb_adapter = ImageNetRGBAdapter()
        self.decoder = build_decoder(
            self.decoder_arch,
            pc_cfg=pc_cfg,
            attach_pc=bool(attach_pc) and self.pc_placement == 'decoder',
        )
        self.encoder_pc_hbm = None
        self.pseudo_refiner = None
        self.encoder_pc_head = None
        self.encoder_pc_config = None
        self.encoder_pc_profile_v3 = False
        if self.pc_placement == 'encoder':
            # The legacy profile registry can still stamp pc_placement onto a
            # v2 config for contract/isolation tests.  Production encoder-PC
            # execution requires the independent strict v3 config.
            self.encoder_pc_profile_v3 = isinstance(pc_cfg, EncoderPCHBMConfig)
            if self.encoder_pc_profile_v3:
                self.encoder_pc_config = pc_cfg
                self.encoder_pc_hbm = EncoderPCHBMAdapter(self.encoder_pc_config)
                self.pseudo_refiner = TeacherPseudoLabelRefiner(
                    self.encoder_pc_config
                )
                # The Adapter/Decoder/Refiner remain registered directly on
                # BaseModel for stable optimizer and checkpoint interfaces.
                # The role head is a policy dispatcher over those same module
                # objects, so it is intentionally not registered a second time.
                object.__setattr__(
                    self,
                    "encoder_pc_head",
                    EncoderPCSegmentationHead(
                        self.encoder_pc_hbm,
                        self.decoder,
                        self.pseudo_refiner,
                    ),
                )

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()
        return self

    @torch.no_grad()
    def extract_feature_bundle(self, x):
        if x.dim() != 4:
            raise ValueError(f'Expected image batch [B,C,H,W], got {tuple(x.shape)}.')
        self.dino.eval()
        layer_indices = (
            tuple(self.pc_cfg.dino_layer_indices)
            if self.pc_cfg is not None
            else (2, 5, 8, 11)
        )
        features = self.dino.get_intermediate_layers(
            x=x,
            n=layer_indices,
            reshape=False,
            return_class_token=True,
            norm=True,
        )
        if len(features) != 4:
            raise RuntimeError(
                f'DINO returned {len(features)} feature levels instead of four.'
            )

        patch_tokens = []
        cls_tokens = []
        for level, feature_pair in enumerate(features, start=1):
            if not isinstance(feature_pair, (tuple, list)) or len(feature_pair) != 2:
                raise RuntimeError(
                    f'DINO level {level} did not return a (patch_tokens, cls_token) pair.'
                )
            patch, cls = feature_pair
            patch_tokens.append(patch)
            cls_tokens.append(cls)
        return DinoFeatureBundle(
            patch_tokens=tuple(patch_tokens),
            cls_tokens=tuple(cls_tokens),
        ).validate()

    @torch.no_grad()
    def extract_features(self, x):
        """Return the historical four-level patch-only DINO interface."""

        return self.extract_feature_bundle(x).patch_tokens

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
        encoder_stage=None,
        allow_memory_fallback=False,
        encoder_role='labeled_core',
        run_labeled_refiner=False,
    ):
        bundle = self.extract_feature_bundle(x)
        x_features = bundle.patch_tokens
        image_rgb = self.prepare_rgb(x)
        if getattr(self, 'pc_placement', 'decoder') == 'encoder':
            encoder_aux = {'mode': 'off', 'pc_active': False}
            if self.encoder_pc_profile_v3:
                if encoder_stage is not None and not isinstance(
                    encoder_stage, EncoderPCStageFlags
                ):
                    raise TypeError('encoder_stage must be EncoderPCStageFlags or None.')
                if run_labeled_refiner and not return_aux:
                    raise ValueError(
                        'run_labeled_refiner requires return_aux=True.'
                    )
                head_result = self.encoder_pc_head(
                    role=encoder_role,
                    bundle=bundle,
                    image_rgb=image_rgb,
                    memory=memory,
                    mode=pc_mode,
                    stage=encoder_stage,
                    epoch=epoch,
                    query_image_ids=query_image_ids,
                    allow_memory_fallback=allow_memory_fallback,
                    return_aux=return_aux or run_labeled_refiner,
                )
                if not isinstance(head_result, EncoderPCCoreResult):
                    return head_result
                if run_labeled_refiner:
                    refiner_output = self.encoder_pc_head(
                        role='labeled_refiner',
                        core_result=head_result,
                        epoch=epoch,
                    )
                    combined_aux = dict(head_result.aux)
                    combined_aux['pseudo_refiner'] = refiner_output
                    head_result = EncoderPCCoreResult(
                        head_result.outputs, combined_aux
                    )
                if return_aux:
                    return head_result.outputs, head_result.aux
                return head_result.outputs
            decoder_result = self.decoder(
                features=x_features,
                image_rgb=image_rgb,
                memory=None,
                pc_mode='off',
                epoch=epoch,
                return_aux=return_aux,
                query_image_ids=None,
            )
            if not return_aux:
                return decoder_result
            if not isinstance(decoder_result, (tuple, list)) or len(decoder_result) != 2:
                raise RuntimeError('BGFBR Decoder must return (outputs, aux).')
            outputs, decoder_aux = decoder_result
            combined_aux = dict(decoder_aux or {})
            combined_aux['encoder_pc_hbm'] = encoder_aux
            combined_aux['pc_active'] = bool(encoder_aux.get('pc_active', False))
            combined_aux['pc_mode'] = str(encoder_aux.get('mode', pc_mode))
            return outputs, combined_aux
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
        bundle = self.extract_feature_bundle(x)
        x_features = bundle.patch_tokens
        image_rgb = self.prepare_rgb(x)
        if self.pc_placement == 'encoder' and self.encoder_pc_profile_v3:
            return self.encoder_pc_head(
                role='inference',
                bundle=bundle,
                image_rgb=image_rgb,
                memory=memory,
                stage=EncoderPCStageFlags(
                    enable_f4_f3=True,
                    f4_f3_progress=1.0,
                    enable_f2_f1=True,
                    f2_f1_progress=1.0,
                ),
                epoch=epoch,
                return_aux=False,
            )
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
        if self.pc_placement == 'encoder':
            raise RuntimeError(
                'encoder_pc must be saved as a complete v3 adapter/decoder/refiner artifact.'
            )
        assert path.endswith('.pth'), f'Path should end with .pth, but got: {path}'
        save_decoder_artifact(path, self.decoder, self.pc_cfg, epoch=0)
        print(f'Successfully save seg parameters to {path}.')

    def load_decoder_checkpoint(self, path, require_pc_complete=False):
        if self.pc_placement == 'encoder':
            raise RuntimeError(
                'encoder_pc accepts only strict non-PC BGFBR warm-start or v3 artifacts.'
            )
        assert path.endswith('.pth'), f'Path should end with .pth, but got: {path}'
        load_decoder_compatible(
            self.decoder,
            path,
            require_pc_complete=bool(require_pc_complete),
        )
        print(f'Successfully load seg parameters from {path}.')
