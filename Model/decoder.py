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

        # PC-HBM is attached in the following implementation stage.  Keeping it
        # ``None`` here preserves the exact legacy state_dict and parity path.
        self.pc_cfg = pc_cfg
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
            'features': {'p1': seg_1_feature},
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

        outputs, aux = self._forward_baseline(features)
        if pc_mode != 'off':
            aux['fallback_reason'] = 'pc_hbm_not_attached'
            aux['forward_mode'] = pc_mode

        if return_aux:
            return outputs, aux
        return outputs
