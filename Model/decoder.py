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
    decoder_arch = 'legacy_transformer'
    decoder_architecture = 'legacy_transformer'
    decoder_contract_version = 1
    VALID_PC_MODES = {'off', 'parent_only', 'full', 'teacher_pseudo', 'student_core'}

    def __init__(self, in_dim=768, out_dim=128, heads=16, hidden_dim=128, dropout=0., pc_cfg=None):
        super().__init__()

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
        bs, patch_num, _ = f_1.shape
        patches = int(math.sqrt(patch_num))
        if patches * patches != patch_num:
            raise ValueError(f'DINO token count must form a square grid, got {patch_num}.')
        seg_res = (patches * 14) // 4

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
            'z_nomix': seg_1,
            'z_final': seg_1,
            'p_final': torch.sigmoid(seg_1),
            'pc_active': False,
            'fallback_reason': None,
            'pc_hbm': None,
            'p2_bra': None,
            'p1_pra': None,
            'mixture': None,
            'mixture_skipped': True,
            'forward_mode': 'off',
            'features': {
                'p3': p3_map,
                'p2': p2_map,
                'p2_pre': p2_map,
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
        m2, _ = self._predict_side(
            t2, torch.sigmoid(m3), self.seg_head_2,
            state['token_hw'], state['output_hw'],
        )
        token_hw = state['token_hw']
        return {
            'x3': tokens_to_map(state['kv3'], *token_hw),
            'p3': tokens_to_map(t3, *token_hw),
            'p2': tokens_to_map(t2, *token_hw) + tokens_to_map(state['kv2'], *token_hw),
            'm3': F.interpolate(m3, size=token_hw, mode='bilinear', align_corners=False),
            'm2': F.interpolate(m2, size=token_hw, mode='bilinear', align_corners=False),
        }

    def _memory_fallback_reason(self, memory):
        if memory is None:
            return 'memory_missing'
        if not hasattr(memory, 'is_ready') or not memory.is_ready():
            return 'memory_not_ready'
        if self.pc_cfg is not None and hasattr(memory, 'validate_compat'):
            try:
                compatible = memory.validate_compat(self.pc_cfg.expected_memory_meta())
            except (KeyError, RuntimeError, ValueError) as error:
                return f'memory_incompatible:{error}'
            if not compatible:
                reason = getattr(compatible, 'reason', None)
                return str(reason or 'memory_incompatible')
        return None

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
        m2_pre_token = F.interpolate(
            m2_pre, size=token_hw, mode='bilinear', align_corners=False
        )

        if pc_mode == 'parent_only':
            pc_aux = self.pc_hbm.forward_parent_only(
                x3=x3_map,
                p3=p3_map,
                m3=m3_token,
                memory=memory,
                query_image_ids=query_image_ids,
            )
            t2 = t2_pre
            m2 = m2_pre
            p2_refined_map = p2_pre_map
            p2_aux = None
        else:
            pc_aux = self.pc_hbm.forward_parent_child(
                x3=x3_map,
                p3=p3_map,
                child_map=child_map,
                m3=m3_token,
                m2_pre=m2_pre_token,
                memory=memory,
                epoch=epoch,
                query_image_ids=query_image_ids,
            )
            t3_corr = map_to_tokens(pc_aux['p3_corr'])
            t2 = self._forward_t2(state, t3_corr)
            m2, _ = self._predict_side(
                t2, torch.sigmoid(m3), self.seg_head_2, token_hw, output_hw
            )
            p2_map = tokens_to_map(t2, *token_hw)
            m2_token = F.interpolate(
                m2, size=token_hw, mode='bilinear', align_corners=False
            )
            p2_aux = self.pc_hbm.forward_p2(
                p2=p2_map,
                prob2=torch.sigmoid(m2_token),
                pc_maps=pc_aux['pc_maps'],
            )
            p2_refined_map = p2_aux['p2_refined']

        t1 = self._forward_t1(state, map_to_tokens(p2_refined_map))
        z_main, p1_98 = self._predict_side(
            t1, torch.sigmoid(m2), self.seg_head_1, token_hw, output_hw
        )
        run_p1 = pc_mode in {'student_core', 'full', 'teacher_pseudo'} and p2_aux is not None
        run_mixture = pc_mode in {'full', 'teacher_pseudo'} and p2_aux is not None
        if run_p1:
            p1_aux = self.pc_hbm.forward_p1(
                p1=p1_98, z_main=z_main, p2_aux=p2_aux
            )
        else:
            p1_aux = None

        if run_mixture:
            mix_aux = self.pc_hbm.forward_mixture(
                z_main=z_main,
                p1_aux=p1_aux,
                pc_maps=pc_aux['pc_maps'],
                epoch=epoch,
                ts_continuation=pc_mode == 'teacher_pseudo',
            )
            z_final = mix_aux['z_final']
            p_final = mix_aux['p_final']
        else:
            mix_aux = None
            if pc_mode == 'student_core':
                z_final = None
                p_final = None
            else:
                z_final = z_main
                p_final = torch.sigmoid(z_main)

        outputs = (m4, m3, m2, z_main, state['global_logit'])
        distill_features = None
        if pc_mode == 'teacher_pseudo':
            distill_features = {
                'p3_corr': pc_aux['p3_corr'],
                'p2_refined': p2_refined_map,
            }
        aux = {
            'm4': m4,
            'm3': m3,
            'm2': m2,
            'global_logit': state['global_logit'],
            'z_main': z_main,
            'z_nomix': z_main,
            'z_final': z_final,
            'p_final': p_final,
            'pc_active': True,
            'fallback_reason': None,
            'pc_hbm': pc_aux,
            'p2_bra': p2_aux,
            'p1_pra': p1_aux,
            'mixture': mix_aux,
            'mixture_skipped': not run_mixture,
            'forward_mode': pc_mode,
            'distill_features': distill_features,
            'features': {
                'x3': x3_map,
                'p3': p3_map,
                'p2_pre': p2_pre_map,
                'p2_refined': p2_refined_map,
                'p1': p1_98,
            },
        }
        return outputs, self.pc_hbm.slim_aux(aux, mode=pc_mode)

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

        fallback_reason = None
        if pc_mode != 'off':
            if self.pc_hbm is None:
                fallback_reason = 'pc_hbm_not_attached'
            else:
                fallback_reason = self._memory_fallback_reason(memory)

        if pc_mode != 'off' and fallback_reason is not None and self.training:
            raise RuntimeError(
                'PC-HBM training requires an attached, ready, compatible memory; '
                f'got {fallback_reason}.'
            )

        if pc_mode == 'off' or fallback_reason is not None:
            outputs, aux = self._forward_baseline(features)
            if pc_mode != 'off':
                aux['fallback_reason'] = fallback_reason
                aux['forward_mode'] = pc_mode
        else:
            outputs, aux = self._forward_pc_hbm(
                features=features,
                memory=memory,
                pc_mode=pc_mode,
                epoch=epoch,
                query_image_ids=query_image_ids,
            )

        if return_aux:
            return outputs, aux
        return outputs


# Historical import compatibility.  There is only one concrete Decoder.
LegacyTransformerDecoder = Decoder


__all__ = [
    'Attention',
    'Decoder',
    'FeedForwardLayer',
    'LegacyTransformerDecoder',
    'TransformerBlock',
    'map_to_tokens',
    'tokens_to_map',
]
