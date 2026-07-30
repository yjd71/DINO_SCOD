import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

def tokens_to_map(tokens, height, width):
    """Convert ``[B, H*W, C]`` tokens to a contiguous feature map."""
    if tokens.dim() != 3:
        raise ValueError(f'Expected token tensor [B,N,C], got {tuple(tokens.shape)}.')
    batch, token_count, channels = tokens.shape
    if token_count != height * width:
        raise ValueError(
            f'Token number {token_count} does not match requested grid {height}x{width}.'
        )
    return tokens.permute(0, 2, 1).reshape(batch, channels, height, width)


def map_to_tokens(feature_map):
    """Convert ``[B,C,H,W]`` maps to contiguous ``[B,H*W,C]`` tokens."""
    if feature_map.dim() != 4:
        raise ValueError(f'Expected feature map [B,C,H,W], got {tuple(feature_map.shape)}.')
    return feature_map.flatten(2).transpose(1, 2).contiguous()


class FeedForwardLayer(nn.Module):
    def __init__(self, dim, hidden_dim=768, dropout=0.):
        super(FeedForwardLayer, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim=768, heads=16, hid_dim=768, dropout=0., use_sdpa=True):
        super(Attention, self).__init__()
        self.heads = heads
        assert hid_dim % heads == 0
        dim_head = hid_dim // heads
        self.scale = dim_head ** -0.5
        self.use_sdpa = use_sdpa  # use SDPA or not
        
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, hid_dim)
        self.to_k = nn.Linear(dim, hid_dim)
        self.to_v = nn.Linear(dim, hid_dim)
        self.to_out = nn.Sequential(nn.Linear(hid_dim, dim), nn.Dropout(dropout))

    def forward(self, q, k, v):
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        q, k, v = map(lambda t: rearrange(t, 'b l (h d) -> b h l d', h=self.heads), (q, k, v))
        
        # use SDPA for faster attention computation
        if self.use_sdpa:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0., is_causal=False)
        else:
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            attn = self.attend(dots)
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
        
        out = rearrange(out, 'b h l d -> b l (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, hidden_dim, dropout=0.):
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, hid_dim=hidden_dim, dropout=dropout)

        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = FeedForwardLayer(dim, hidden_dim=dim, dropout=dropout)

    def forward(self, q, kv):
        x = kv + self.attn(q, kv, kv)
        x = self.attn_norm(x)
        x = x + self.ffn(x)
        out = self.ffn_norm(x)

        return out


class Decoder(nn.Module):
    VALID_PC_MODES = {'off', 'verify_only', 'full', 'teacher_pseudo'}

    def __init__(self, in_dim=768, out_dim=128, heads=16, hidden_dim=128, dropout=0., pc_cfg=None):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        if (
            pc_cfg is not None
            and getattr(pc_cfg, 'enabled', False)
            and (self.in_dim, self.out_dim) != (768, 128)
        ):
            raise ValueError(
                'PC-HBM-Lite requires Decoder in_dim=768 and out_dim=128.'
            )

        self.linear_1 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )
        self.linear_2 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )
        self.linear_3 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )
        self.linear_4 = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )

        self.linear_1234 = nn.Sequential(
            nn.Linear(in_dim * 4, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )
        
        self.linear_34 = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )
        self.linear_23 = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )
        self.linear_12 = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim)
        )

        self.TransBlock_seg1 = TransformerBlock(out_dim, heads, hidden_dim, dropout=dropout)
        self.TransBlock_seg2 = TransformerBlock(out_dim, heads, hidden_dim, dropout=dropout)
        self.TransBlock_seg3 = TransformerBlock(out_dim, heads, hidden_dim, dropout=dropout)
        self.TransBlock_seg4 = TransformerBlock(out_dim, heads, hidden_dim, dropout=dropout)

        self.seg_global = nn.Conv2d(in_channels=out_dim, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        self.seg_head_1 = nn.Conv2d(in_channels=out_dim * 2, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.seg_head_2 = nn.Conv2d(in_channels=out_dim * 2, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.seg_head_3 = nn.Conv2d(in_channels=out_dim * 2, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.seg_head_4 = nn.Conv2d(in_channels=out_dim * 2, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)

        self.pc_cfg = pc_cfg
        if pc_cfg is not None and getattr(pc_cfg, 'enabled', False):
            from Model.PC_HBM.dino_engine import DinoPCHBMEngine

            self.pc_hbm = DinoPCHBMEngine(pc_cfg)
        else:
            self.pc_hbm = None

    def _project_features(self, features):
        if not isinstance(features, (tuple, list)) or len(features) != 4:
            raise ValueError('Decoder expects exactly four DINO feature tensors.')
        f_1, f_2, f_3, f_4 = features
        if not all(torch.is_tensor(feature) for feature in features):
            raise TypeError('Every DINO feature level must be a tensor.')
        if any(feature.dim() != 3 for feature in features):
            raise ValueError('Every DINO feature level must be [B,N,C].')
        if any(feature.shape != f_1.shape for feature in features[1:]):
            raise ValueError('All four DINO feature levels must share one shape.')
        if any(
            feature.device != f_1.device or feature.dtype != f_1.dtype
            for feature in features[1:]
        ):
            raise ValueError(
                'All four DINO feature levels must share device and dtype.'
            )
        if f_1.size(-1) != self.in_dim:
            raise ValueError(
                f'DINO feature width must be {self.in_dim}, got {f_1.size(-1)}.'
            )
        bs, patch_num, _ = f_1.shape
        patches = int(math.sqrt(patch_num))
        if patches * patches != patch_num:
            raise ValueError(f'DINO token count must form a square grid, got {patch_num}.')
        if (
            self.pc_cfg is not None
            and getattr(self.pc_cfg, 'enabled', False)
            and patches != 28
        ):
            raise ValueError(
                f'PC-HBM-Lite requires a 28x28 DINO grid, got {patches}x{patches}.'
            )
        seg_res = (patches * 14) // 4
        if (
            self.pc_cfg is not None
            and getattr(self.pc_cfg, 'enabled', False)
            and seg_res != 98
        ):
            raise ValueError(
                f'PC-HBM-Lite requires 98x98 output, got {seg_res}x{seg_res}.'
            )

        query = self.linear_1234(torch.cat([f_1, f_2, f_3, f_4], dim=-1))
        global_mask = tokens_to_map(query, patches, patches)
        global_mask = F.interpolate(global_mask, size=(seg_res, seg_res), mode='bilinear', align_corners=False)
        global_mask = self.seg_global(global_mask)
        kv_1 = self.linear_1(f_1)
        kv_2 = self.linear_2(f_2)
        kv_3 = self.linear_3(f_3)
        kv_4 = self.linear_4(f_4)

        return {
            'batch_size': bs,
            'token_hw': (patches, patches),
            'output_hw': (seg_res, seg_res),
            'query': query,
            'kv1': kv_1,
            'kv2': kv_2,
            'kv3': kv_3,
            'kv4': kv_4,
            'global_logit': global_mask,
            'global_mask': torch.sigmoid(global_mask),
        }

    def _forward_t4(self, state):
        return self.TransBlock_seg4(q=state['query'], kv=state['kv4'])

    def _forward_t3(self, state, t4):
        kv = self.linear_34(torch.cat([state['kv3'], t4], dim=-1))
        return self.TransBlock_seg3(q=state['query'], kv=kv)

    def _forward_t2(self, state, t3):
        kv = self.linear_23(torch.cat([state['kv2'], t3], dim=-1))
        return self.TransBlock_seg2(q=state['query'], kv=kv)

    def _forward_t1(self, state, t2):
        kv = self.linear_12(torch.cat([state['kv1'], t2], dim=-1))
        return self.TransBlock_seg1(q=state['query'], kv=kv)

    @staticmethod
    def _tokens_at_output_scale(tokens, token_hw, output_hw):
        feature_map = tokens_to_map(tokens, *token_hw)
        return F.interpolate(feature_map, size=output_hw, mode='bilinear', align_corners=False)

    def _predict_side(self, tokens, previous_mask, head, token_hw, output_hw):
        feature_map = self._tokens_at_output_scale(tokens, token_hw, output_hw)
        logit = head(torch.cat([feature_map, feature_map * previous_mask], dim=1))
        return logit, feature_map

    def _forward_baseline(self, features):
        state = self._project_features(features)

        seg_4_tokens = self._forward_t4(state)
        seg_3_tokens = self._forward_t3(state, seg_4_tokens)
        seg_2_tokens = self._forward_t2(state, seg_3_tokens)
        seg_1_tokens = self._forward_t1(state, seg_2_tokens)

        # Keep the baseline numerical path unchanged while exposing the raw
        # 28x28 decoder features used by teacher-only feature distillation.
        p3_map = tokens_to_map(seg_3_tokens, *state['token_hw'])
        p2_map = tokens_to_map(seg_2_tokens, *state['token_hw'])

        seg_4, _ = self._predict_side(
            seg_4_tokens, state['global_mask'], self.seg_head_4,
            state['token_hw'], state['output_hw'],
        )
        mask_4 = torch.sigmoid(seg_4)
        seg_3, _ = self._predict_side(
            seg_3_tokens, mask_4, self.seg_head_3,
            state['token_hw'], state['output_hw'],
        )
        mask_3 = torch.sigmoid(seg_3)
        seg_2, _ = self._predict_side(
            seg_2_tokens, mask_3, self.seg_head_2,
            state['token_hw'], state['output_hw'],
        )
        mask_2 = torch.sigmoid(seg_2)
        seg_1, seg_1_feature = self._predict_side(
            seg_1_tokens, mask_2, self.seg_head_1,
            state['token_hw'], state['output_hw'],
        )

        outputs = (seg_4, seg_3, seg_2, seg_1, state['global_logit'])
        aux = {
            'm4': seg_4,
            'm3': seg_3,
            'm2': seg_2,
            'global_logit': state['global_logit'],
            'z_main': seg_1,
            'z_final': seg_1,
            'p_final': torch.sigmoid(seg_1),
            'pc_active': False,
            'fallback_reason': None,
            'pc_hbm': None,
            'forward_mode': 'off',
            'features': {
                'p3': p3_map,
                'p2': p2_map,
                'p1': seg_1_feature,
            },
            'distill_features': None,
        }
        return outputs, aux

    @torch.no_grad()
    def forward_memory_features(self, features):
        state = self._project_features(features)
        t4 = self._forward_t4(state)
        m4, _ = self._predict_side(
            t4, state['global_mask'], self.seg_head_4,
            state['token_hw'], state['output_hw'],
        )
        t3 = self._forward_t3(state, t4)
        m3, _ = self._predict_side(
            t3, torch.sigmoid(m4), self.seg_head_3,
            state['token_hw'], state['output_hw'],
        )
        t2 = self._forward_t2(state, t3)
        token_hw = state['token_hw']
        return {
            'x3': tokens_to_map(state['kv3'], *token_hw),
            'p3': tokens_to_map(t3, *token_hw),
            'p2': tokens_to_map(t2, *token_hw) + tokens_to_map(state['kv2'], *token_hw),
            'm3': F.interpolate(m3, size=token_hw, mode='bilinear', align_corners=False),
        }

    def _validate_memory(self, memory):
        """Reject any provided non-V2/incompatible memory before computation."""

        if not hasattr(memory, 'is_ready') or not callable(memory.is_ready):
            raise TypeError('PC-HBM memory must expose is_ready().')
        if not memory.is_ready():
            raise ValueError('PC-HBM memory is not finalized and ready.')
        if not hasattr(memory, 'validate_compat') or not callable(memory.validate_compat):
            raise TypeError('PC-HBM memory must expose validate_compat().')
        compatible = memory.validate_compat(self.pc_cfg.expected_memory_meta())
        if not compatible:
            reason = getattr(compatible, 'reason', None)
            raise ValueError(
                f'Incompatible PC-HBM-Lite memory: {reason or "unknown_reason"}'
            )

    def _forward_pc_hbm(
        self,
        features,
        memory,
        pc_mode,
        epoch,
        query_image_ids=None,
    ):
        state = self._project_features(features)
        token_hw = state['token_hw']
        output_hw = state['output_hw']

        t4 = self._forward_t4(state)
        m4, _ = self._predict_side(
            t4, state['global_mask'], self.seg_head_4, token_hw, output_hw
        )
        t3 = self._forward_t3(state, t4)
        m3, _ = self._predict_side(
            t3, torch.sigmoid(m4), self.seg_head_3, token_hw, output_hw
        )
        t2_pre = self._forward_t2(state, t3)
        m2_pre, _ = self._predict_side(
            t2_pre, torch.sigmoid(m3), self.seg_head_2, token_hw, output_hw
        )

        x3_map = tokens_to_map(state['kv3'], *token_hw)
        p3_map = tokens_to_map(t3, *token_hw)
        p2_pre_map = tokens_to_map(t2_pre, *token_hw)
        kv2_map = tokens_to_map(state['kv2'], *token_hw)
        child_map = p2_pre_map + kv2_map
        m3_token = F.interpolate(
            m3, size=token_hw, mode='bilinear', align_corners=False
        )

        if pc_mode == 'verify_only':
            injection_scale = 0.0
        elif pc_mode == 'teacher_pseudo':
            injection_scale = 1.0
        elif epoch is None:
            injection_scale = 1.0
        else:
            injection_scale = float(self.pc_cfg.injection_scale(int(epoch)))
        pc_aux = self.pc_hbm.forward_lite(
            x3=x3_map,
            p3=p3_map,
            p2=child_map,
            m3=m3_token,
            memory=memory,
            mode=pc_mode,
            injection_scale=injection_scale,
            query_image_ids=query_image_ids,
        )

        if pc_mode == 'verify_only':
            t2 = t2_pre
            m2 = m2_pre
        else:
            t3_corr = map_to_tokens(pc_aux['p3_corr'])
            t2 = self._forward_t2(state, t3_corr)
            m2, _ = self._predict_side(
                t2, torch.sigmoid(m3), self.seg_head_2, token_hw, output_hw
            )

        p2_map = tokens_to_map(t2, *token_hw)
        t1 = self._forward_t1(state, t2)
        z_main, p1_98 = self._predict_side(
            t1, torch.sigmoid(m2), self.seg_head_1, token_hw, output_hw
        )
        z_final = z_main
        p_final = torch.sigmoid(z_main)

        outputs = (m4, m3, m2, z_main, state['global_logit'])
        distill_features = None
        if pc_mode == 'teacher_pseudo':
            distill_features = {
                'p3_corr': pc_aux['p3_corr'],
            }
        aux = {
            'm4': m4,
            'm3': m3,
            'm2': m2,
            'global_logit': state['global_logit'],
            'z_main': z_main,
            'z_final': z_final,
            'p_final': p_final,
            'pc_active': True,
            'fallback_reason': None,
            'pc_hbm': pc_aux,
            'forward_mode': pc_mode,
            'distill_features': distill_features,
            'features': {
                'p3': p3_map,
                'p2': p2_map,
                'p1': p1_98,
            },
        }
        return outputs, aux

    def forward(
        self,
        features,
        memory=None,
        pc_mode='off',
        epoch=None,
        return_aux=False,
        query_image_ids=None,
    ):
        if pc_mode not in self.VALID_PC_MODES:
            raise ValueError(f'Unsupported pc_mode={pc_mode!r}. Expected one of {sorted(self.VALID_PC_MODES)}.')

        if memory is not None:
            if self.pc_hbm is None:
                raise RuntimeError(
                    'A memory was provided but this Decoder has no PC-HBM-Lite engine.'
                )
            self._validate_memory(memory)

        if pc_mode == 'off':
            outputs, aux = self._forward_baseline(features)
        elif memory is None:
            outputs, aux = self._forward_baseline(features)
            aux['fallback_reason'] = 'memory_missing'
            aux['forward_mode'] = pc_mode
        else:
            outputs, aux = self._forward_pc_hbm(
                features=features,
                memory=memory,
                pc_mode=pc_mode,
                epoch=epoch,
                query_image_ids=query_image_ids,
            )

        if pc_mode != 'off' and memory is None:
            if self.pc_hbm is None:
                aux['fallback_reason'] = 'pc_hbm_not_attached'
                aux['forward_mode'] = pc_mode

        if return_aux:
            return outputs, aux
        return outputs
