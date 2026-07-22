"""DINOv2 same-grid orchestration for PC-HBM."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .common.utils import gather_tokens, merge_parent_results
from .dino_channel_spec import DinoPCHBMChannelSpec
from .dino_memory_builder import DinoMemoryBuilder
from .fusion.hypothesis_token_builder import HypothesisTokenBuilder
from .fusion.p3_gated_residual import P3GatedResidual
from .fusion.pc_hca import PCHCA
from .fusion.pc_scatter import pc_scatter
from .fusion.pc_token_decoder import PCTokenDecoder
from .fusion.query_state_builder import QueryStateBuilder
from .fusion.structured_gate_mlp import StructuredGateMLP
from .refinement.adaptive_mixture_head import AdaptiveMixtureHead
from .refinement.boundary_query_head import BoundaryQueryHead3
from .refinement.p1_pixel_refinement_attention import P1PixelRefinementAttention
from .refinement.p2_boundary_retarget_attention import P2BoundaryRetargetAttention
from .retrieval.child_query_builder import DinoChildQueryBuilder
from .retrieval.child_verifier_v2 import ChildVerifierV2
from .retrieval.parent_retriever import ParentRetriever
from .routing.camouflage_context_router import CamouflageContextRouter


def _boundary_features(probability: torch.Tensor) -> torch.Tensor:
    probability = probability.clamp(1e-6, 1.0 - 1e-6)
    dilated = F.max_pool2d(probability, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-probability, kernel_size=3, stride=1, padding=1)
    morphology = (dilated - eroded).clamp(0.0, 1.0)
    uncertainty = 4.0 * probability * (1.0 - probability)
    dx = F.pad(probability[..., :, 1:] - probability[..., :, :-1], (0, 1, 0, 0))
    dy = F.pad(probability[..., 1:, :] - probability[..., :-1, :], (0, 0, 0, 1))
    gradient = torch.sqrt(dx.square() + dy.square() + 1e-6).clamp(0.0, 1.0)
    entropy = -(
        probability * torch.log(probability)
        + (1.0 - probability) * torch.log(1.0 - probability)
    ) / 0.6931471805599453
    return torch.cat([morphology, uncertainty, gradient, entropy, probability], dim=1)


class DinoPCHBMEngine(nn.Module):
    """Compose route, hypothesis verification and hierarchical refinements."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        spec = DinoPCHBMChannelSpec(
            x3=cfg.decoder_dim,
            p3=cfg.decoder_dim,
            p2=cfg.decoder_dim,
            p1=cfg.decoder_dim,
            pc_dim=cfg.memory_dim,
            value_dim=cfg.value_dim,
            geometry_dim=cfg.geometry_dim,
        )
        dim = cfg.memory_dim
        boundary_contract = tuple(
            getattr(cfg, 'boundary_feature_channels', (5, 8, 8, 14))
        )
        if boundary_contract != (5, 8, 8, 14):
            raise ValueError(
                'boundary_feature_channels is fixed to (5, 8, 8, 14) '
                'for the original Decoder'
            )
        self.p3_boundary_in_ch = 5

        self.boundary3 = BoundaryQueryHead3(
            top_ratio=cfg.p3_top_ratio,
            min_tokens=cfg.p3_min_tokens,
            max_tokens=cfg.p3_max_tokens,
            in_ch=self.p3_boundary_in_ch,
        )
        self.router = CamouflageContextRouter(
            x3_ch=spec.x3, dim=dim, top_img_k=cfg.route_top_img_k
        )
        self.parent_retriever = ParentRetriever(
            p3_ch=spec.p3,
            dim=dim,
            topk=cfg.parent_topk,
            tau=cfg.tau_parent,
        )
        self.child_query = DinoChildQueryBuilder(
            p2_ch=spec.p2, dim=dim, window=cfg.child_window_size
        )
        self.child_verifier = ChildVerifierV2(
            dim=dim,
            value_dim=cfg.value_dim,
            geometry_dim=cfg.geometry_dim,
            tau=cfg.tau_child,
        )
        self.hyp_builder = HypothesisTokenBuilder(
            dim=dim,
            value_dim=cfg.value_dim,
            geometry_dim=cfg.geometry_dim,
        )
        self.query_state = QueryStateBuilder(dim=dim)
        self.hca = PCHCA(
            dim=dim,
            num_heads=cfg.attn_num_heads,
            head_dim=cfg.attn_head_dim,
            tau=cfg.tau_hca,
        )
        self.token_decoder = PCTokenDecoder(
            dim=dim, value_dim=cfg.value_dim, geometry_dim=cfg.geometry_dim
        )
        self.gate_mlp = StructuredGateMLP()
        self.p3_residual = P3GatedResidual(dim=dim, p3_ch=spec.p3)
        self.p2_bra = P2BoundaryRetargetAttention(
            p2_ch=spec.p2,
            dim=dim,
            window=cfg.p2_local_window,
            tau=cfg.tau_bra,
            top_ratio=cfg.p2_top_ratio,
            min_tokens=cfg.p2_min_tokens,
            max_tokens=cfg.p2_max_tokens,
            detach_refs=cfg.detach_p3_refs_for_p2,
            num_heads=cfg.attn_num_heads,
            head_dim=cfg.attn_head_dim,
            boundary_in_ch=8,
        )
        self.p1_pra = P1PixelRefinementAttention(
            p1_ch=spec.p1,
            dim=dim,
            window=cfg.p1_local_window,
            tau=cfg.tau_pra,
            top_ratio=cfg.p1_top_ratio,
            min_tokens=cfg.p1_min_tokens,
            max_tokens=cfg.p1_max_tokens,
            detach_refs=cfg.detach_p2_refs_for_p1,
            num_heads=cfg.attn_num_heads,
            head_dim=cfg.attn_head_dim,
            boundary_in_ch=8,
        )
        self.mixture = AdaptiveMixtureHead(
            r_max=cfg.r_max,
            max_offset=cfg.max_offset,
            mask_corr_epsilon=cfg.mask_corr_epsilon,
            init_bias=cfg.mixture_init_bias,
            use_branch_quality=True,
            use_branch_dropout=True,
            context_ch=14,
        )
        self.memory_builder = DinoMemoryBuilder(
            cfg, self.router, self.parent_retriever, self.child_query
        )

    @staticmethod
    def _selected(index_dict: Dict[str, torch.Tensor], key: str) -> torch.Tensor:
        if key not in index_dict:
            raise KeyError(f'Boundary index dictionary is missing {key!r}.')
        return index_dict[key]

    def _empty_parent_result(self, reference: torch.Tensor) -> Dict[str, object]:
        k = self.cfg.parent_topk
        dim = self.cfg.memory_dim
        value_dim = self.cfg.value_dim
        geo_dim = self.cfg.geometry_dim
        return {
            'q3': reference.new_empty((0, dim)),
            'top_parent_keys': reference.new_empty((0, k, dim)),
            'top_parent_values': reference.new_empty((0, k, value_dim)),
            'top_parent_geo': reference.new_empty((0, k, geo_dim)),
            'top_child_ptrs': torch.empty((0, k), device=reference.device, dtype=torch.long),
            'top_parent_indices': torch.empty((0, k), device=reference.device, dtype=torch.long),
            'top_parent_scores': reference.new_empty((0, k)),
            'top_parent_valid': torch.empty((0, k), device=reference.device, dtype=torch.bool),
            'A_parent': reference.new_empty((0, k)),
            'P3_group': reference.new_empty((0, 4)),
            'S_fg_parent': reference.new_empty((0, 1)),
            'S_bg_parent': reference.new_empty((0, 1)),
            'M_parent': reference.new_empty((0, 1)),
            'parent_entropy': reference.new_empty((0,)),
            'top_parent_region_ids': torch.empty((0, k), device=reference.device, dtype=torch.long),
            'top_parent_reliability': reference.new_empty((0, k)),
            'top_parent_meta': [],
        }

    def _routed_parent_retrieval(
        self,
        p3: torch.Tensor,
        batch_ids: torch.Tensor,
        flat_indices: torch.Tensor,
        route: Dict[str, object],
        memory,
        query_image_ids: Optional[Sequence[str]],
    ) -> Dict[str, object]:
        q_map = self.parent_retriever.encode_q_map(p3)
        q3 = gather_tokens(q_map, batch_ids, flat_indices)
        if q3.shape[0] == 0:
            return self._empty_parent_result(p3)

        results = []
        for batch_index in range(p3.shape[0]):
            output_positions = torch.where(batch_ids == batch_index)[0]
            if output_positions.numel() == 0:
                continue
            q_b = q3.index_select(0, output_positions)
            exclude_id = None
            if self.cfg.exclude_self_match and query_image_ids is not None:
                exclude_id = str(query_image_ids[batch_index])
            bank_b = memory.get_parent_subbank(
                route['top_img_ids'][batch_index],
                exclude_image_id=exclude_id,
                device=p3.device,
                dtype=p3.dtype,
            )
            result = self.parent_retriever.retrieve_q(
                q_b, bank_b, chunk_size=self.cfg.query_chunk_size
            )
            result['output_positions'] = output_positions
            results.append(result)

        merged = merge_parent_results(results, total_queries=q3.shape[0])
        merged['q3'] = q3
        return merged

    def forward_parent_only(
        self,
        x3: torch.Tensor,
        p3: torch.Tensor,
        m3: torch.Tensor,
        memory,
        query_image_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        prob3 = torch.sigmoid(m3)
        boundary_input = _boundary_features(prob3)
        if boundary_input.size(1) != self.p3_boundary_in_ch:
            raise RuntimeError(
                f'P3 boundary contract expected {self.p3_boundary_in_ch} channels, '
                f'got {boundary_input.size(1)}'
            )
        b3, boundary_indices = self.boundary3(boundary_input)
        batch_ids = self._selected(boundary_indices, 'batch_ids')
        flat_indices = self._selected(boundary_indices, 'flat_indices')
        route = self.router(
            x3,
            prob3,
            memory,
            top_img_k=self.cfg.route_top_img_k,
            query_image_ids=query_image_ids,
            exclude_self_match=self.cfg.exclude_self_match,
        )
        parent_ret = self._routed_parent_retrieval(
            p3, batch_ids, flat_indices, route, memory, query_image_ids
        )
        route_context = route['route_embed'].index_select(0, batch_ids)
        uncertainty_map = 4.0 * prob3 * (1.0 - prob3)
        uncertainty = gather_tokens(uncertainty_map, batch_ids, flat_indices)
        token_scores = self._selected(boundary_indices, 'token_scores').reshape(-1, 1)
        return {
            'B3': b3,
            'boundary_indices3': boundary_indices,
            'batch_ids3': batch_ids,
            'flat_indices3': flat_indices,
            'boundary_confidence': token_scores,
            'uncertainty_token': uncertainty,
            'route': route,
            'route_context_token': route_context,
            'route_entropy': route.get('route_entropy'),
            'route_entropy_norm': route.get('route_entropy_norm'),
            'parent_ret': parent_ret,
            'parent_entropy': parent_ret['parent_entropy'],
            'query_valid': parent_ret['top_parent_valid'].any(dim=1),
        }

    def forward_parent_child(
        self,
        x3: torch.Tensor,
        p3: torch.Tensor,
        child_map: torch.Tensor,
        m3: torch.Tensor,
        m2_pre: torch.Tensor,
        memory,
        epoch: Optional[int],
        query_image_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        aux = self.forward_parent_only(
            x3,
            p3,
            m3,
            memory,
            query_image_ids,
        )
        parent_ret = aux['parent_ret']
        batch_ids = aux['batch_ids3']
        flat_indices = aux['flat_indices3']
        query_valid = aux['query_valid']
        child_query = self.child_query(
            child_map,
            m2_pre,
            batch_ids,
            flat_indices,
            p3_hw=p3.shape[-2:],
        )
        child_bank = memory.get_child_by_ptr(
            parent_ret['top_child_ptrs'],
            device=p3.device,
            dtype=p3.dtype,
            valid_mask=parent_ret['top_parent_valid'],
        )
        child_ver = self.child_verifier(
            child_query['q_child'],
            child_query['G2_query'],
            parent_ret,
            child_bank,
        )
        candidate_valid = child_ver['top_parent_valid']
        query_valid = candidate_valid.any(dim=1)
        hypothesis = self.hyp_builder(
            parent_ret,
            child_ver,
            top_parent_valid=candidate_valid,
            query_valid=query_valid,
        )
        q_state = self.query_state(
            parent_ret['q3'],
            child_query['q_child'],
            aux['route_context_token'],
            child_ver['C23_token'],
            parent_ret['parent_entropy'],
            query_valid=query_valid,
        )
        q_new, attn = self.hca(
            q_state,
            hypothesis,
            child_ver['prior_bias'],
            aux['route_context_token'],
            mask=candidate_valid,
            query_valid=query_valid,
        )
        token_aux = self.token_decoder(
            q_new,
            attn,
            parent_ret,
            child_ver,
            top_parent_valid=candidate_valid,
            query_valid=query_valid,
        )
        gate_pc = self.gate_mlp(
            aux['boundary_confidence'],
            child_ver['C23_token'],
            aux['uncertainty_token'],
            parent_ret['parent_entropy'],
            child_ver['child_entropy'],
            child_ver['S_child'],
            child_ver['S_geo'],
            top_parent_valid=candidate_valid,
            query_valid=query_valid,
        )
        injection_scale = self.cfg.injection_scale(epoch or self.cfg.full_pc_start_epoch)
        p3_corr, p3_delta = self.p3_residual(
            p3,
            batch_ids,
            flat_indices,
            token_aux['Z3_token'],
            gate=injection_scale,
            gate_pc=gate_pc,
            query_valid=query_valid,
        )
        pc_maps = pc_scatter(
            batch_size=p3.shape[0],
            height=p3.shape[-2],
            width=p3.shape[-1],
            batch_ids=batch_ids,
            flat_indices=flat_indices,
            token_aux=token_aux,
            gate_pc_token=gate_pc,
            c23_token=child_ver['C23_token'],
            query_valid=query_valid,
        )
        aux.update(
            {
                'child_query': child_query,
                'child_ver': child_ver,
                'hypothesis_tokens': hypothesis,
                'q_state': q_state,
                'q_new': q_new,
                'pc_attention': attn,
                'token_aux': token_aux,
                'gate_pc_token': gate_pc,
                'query_valid': query_valid,
                'p3_delta': p3_delta,
                'p3_corr': p3_corr,
                'pc_maps': pc_maps,
                'C23_map': pc_maps['C23_map'],
                'gate_pc_map': pc_maps['gate_pc_map'],
                'injection_scale': injection_scale,
            }
        )
        return aux

    def forward_p2(
        self,
        p2: torch.Tensor,
        prob2: torch.Tensor,
        pc_maps: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return self.p2_bra(
            p2=p2,
            prob2=prob2,
            pc_maps=pc_maps,
        )

    def forward_p1(
        self,
        p1: torch.Tensor,
        z_main: torch.Tensor,
        p2_aux: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        return self.p1_pra(
            p1=p1,
            z_main=z_main,
            p2_aux=p2_aux,
        )

    def forward_mixture(
        self,
        z_main: torch.Tensor,
        p1_aux: Dict[str, torch.Tensor],
        pc_maps: Dict[str, torch.Tensor],
        epoch: Optional[int],
        *,
        ts_continuation: bool = False,
    ) -> Dict[str, torch.Tensor]:
        temperature, epsilon = self.cfg.mixture_schedule(
            epoch, ts_continuation=ts_continuation
        )
        return self.mixture(
            z_main=z_main,
            p1_aux=p1_aux,
            pc_maps=pc_maps,
            epoch=epoch,
            temperature=temperature,
            eps_floor=epsilon,
        )

    @torch.no_grad()
    def build_memory_entries(
        self,
        features: Dict[str, torch.Tensor],
        gt: torch.Tensor,
        image_ids: Sequence[str],
    ) -> Dict[str, Dict[str, object]]:
        return self.memory_builder(features, gt, image_ids)

    @staticmethod
    def slim_aux(aux: Dict[str, object], mode: str) -> Dict[str, object]:
        if mode == 'full' or mode == 'parent_only':
            return aux
        slim = dict(aux)
        slim.pop('features', None)
        pc_aux = slim.get('pc_hbm')
        if isinstance(pc_aux, dict):
            if mode == 'student_core':
                keep = {'p3_corr'}
            else:
                keep = {
                    'C23_map', 'gate_pc_map', 'route_entropy', 'route_entropy_norm',
                    'pc_maps', 'injection_scale', 'query_valid',
                }
            slim['pc_hbm'] = {key: value for key, value in pc_aux.items() if key in keep}
        if mode == 'student_core':
            p2_aux = slim.get('p2_bra')
            if isinstance(p2_aux, dict):
                slim['p2_bra'] = {
                    key: value
                    for key, value in p2_aux.items()
                    if key == 'p2_refined'
                }
            p1_aux = slim.get('p1_pra')
            if isinstance(p1_aux, dict):
                p1_keep = {
                    'B1', 'G1_raw_map', 'R1_map', 'O1_map', 'R_sup_map',
                    'valid1_map',
                }
                slim['p1_pra'] = {
                    key: value for key, value in p1_aux.items() if key in p1_keep
                }
        mixture = slim.get('mixture')
        if isinstance(mixture, dict) and mode == 'teacher_pseudo':
            keep = {'pi', 'z_final', 'p_final', 'Mask_corr'}
            slim['mixture'] = {key: value for key, value in mixture.items() if key in keep}
        return slim
