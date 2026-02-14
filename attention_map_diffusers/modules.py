import math
import inspect
import numpy as np
from typing import Any, Dict, Optional, Tuple, Union, List, Callable

import torch
import torch.nn.functional as F
from einops import rearrange

from diffusers.models.attention import _chunked_feed_forward
from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.transformers.transformer_2d import Transformer2DModelOutput
from diffusers.pipelines.flux.pipeline_flux import FluxPipelineOutput
from diffusers.utils import deprecate, logging, USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from diffusers.models.attention_processor import Attention, AttnProcessor, AttnProcessor2_0

logger = logging.get_logger(__name__)
attn_maps = {}

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool: attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else: attn_bias += attn_mask.to(query.dtype)
    attn_weight = torch.softmax(query @ key.transpose(-2, -1) * scale_factor + attn_bias.to(query.device), dim=-1)
    return torch.dropout(attn_weight, dropout_p, train=True) @ value, attn_weight

def sparsify_top_k(attn_probs, k):
    if k <= 0 or k >= attn_probs.shape[-1]: return attn_probs
    topk_values, topk_indices = torch.topk(attn_probs, k, dim=-1)
    mask = torch.zeros_like(attn_probs).scatter_(-1, topk_indices, 1.0)
    sparsified = attn_probs * mask
    return sparsified / (sparsified.sum(dim=-1, keepdim=True) + 1e-10)

def joint_attn_call2_0(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, height=None, timestep=None, *args, **kwargs) -> torch.FloatTensor:
    residual, batch_size = hidden_states, hidden_states.shape[0]
    
    from .utils import flop_counter, cache_schedule, top_k_k
    # Safety check for timestep
    if timestep is not None:
        ts_val = str(int(timestep[0].item())) if torch.is_tensor(timestep) else str(int(timestep))
    else:
        ts_val = None
        
    layer_name = getattr(self, "layer_name", None)
    config = cache_schedule.get(ts_val, {}).get(layer_name, {}) if ts_val else {}
    use_cache, use_top_k = (config.get("cache", False), config.get("top_k", False)) if isinstance(config, dict) else (config, False)

    # OUTPUT CACHING (Strategy 3: Full Skip)
    if use_cache and not use_top_k and hasattr(self, "prev_attn_output"):
        hidden_states = self.prev_attn_output
        encoder_hidden_states = getattr(self, "prev_context_attn_output", None)
    else:
        # We need V for Rigorous Skip and Full Compute
        value = attn.to_v(hidden_states)
        inner_dim, head_dim = value.shape[-1], value.shape[-1] // attn.heads
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if encoder_hidden_states is not None:
            ev = attn.add_v_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = torch.cat([value, ev], dim=2)

        # RIGOROUS SKIP (Strategy 4: Map Cache + Current V + Top-K)
        if use_cache and use_top_k and hasattr(self, "prev_attn_probs"):
            attention_probs = self.prev_attn_probs
            Lq, Lk = attention_probs.shape[2], attention_probs.shape[3]
            actual_k = min(top_k_k, Lk) if top_k_k > 0 else Lk
            flop_counter.add_flops("attention_prob_v", 2 * batch_size * attn.heads * Lq * actual_k * head_dim)
            if top_k_k > 0: attention_probs = sparsify_top_k(attention_probs, top_k_k)
            hidden_states = torch.matmul(attention_probs, value)
        
        # FULL COMPUTE OR PURE TOP-K (Strategy 1 & 2)
        else:
            query, key = attn.to_q(hidden_states), attn.to_k(hidden_states)
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_q is not None: query = attn.norm_q(query)
            if attn.norm_k is not None: key = attn.norm_k(key)
            if encoder_hidden_states is not None:
                eq, ek = attn.add_q_proj(encoder_hidden_states), attn.add_k_proj(encoder_hidden_states)
                eq = eq.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                ek = ek.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
                if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
                query, key = torch.cat([query, eq], dim=2), torch.cat([key, ek], dim=2)

            Lq, Lk = query.shape[2], key.shape[2]
            flop_counter.add_flops("attention_q_k", 2 * batch_size * attn.heads * Lq * Lk * head_dim)
            flop_counter.add_flops("attention_softmax", 3 * batch_size * attn.heads * Lq * Lk)
            actual_k = min(top_k_k, Lk) if (use_top_k and top_k_k > 0) else Lk
            flop_counter.add_flops("attention_prob_v", 2 * batch_size * attn.heads * Lq * actual_k * head_dim)

            hidden_states, attention_probs = scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
            if use_top_k and top_k_k > 0: attention_probs = sparsify_top_k(attention_probs, top_k_k)
            hidden_states = torch.matmul(attention_probs, value)
            
            if ts_val and layer_name:
                self.prev_attn_probs = attention_probs.clone(); self.prev_attn_output = hidden_states

        # Stats Collection
        collect_cross, collect_self = getattr(self, "collect_cross_attn", False), getattr(self, "collect_self_attn", False)
        if (collect_cross or collect_self) and encoder_hidden_states is not None:
            image_length = Lq - (eq.shape[2] if 'eq' in locals() else 0)
            if collect_cross:
                ac = attention_probs[:, :, :image_length, image_length:].cpu()
                if hasattr(self, 'prev_attn_map'): self.similarity = F.cosine_similarity(ac.flatten(1), self.prev_attn_map.flatten(1), dim=1).mean().item()
                else: self.similarity = 0.0
                self.entropy = -(ac * torch.log(ac + 1e-10)).sum(dim=-1).mean().item()
                self.prev_attn_map, self.attn_map = ac.clone(), rearrange(ac, 'b h (ht w) d -> b h ht w d', ht=height)
            if collect_self:
                aself = attention_probs[:, :, :image_length, :image_length].cpu()
                if hasattr(self, 'prev_self_attn_map'): self.self_similarity = F.cosine_similarity(aself.flatten(1), self.prev_self_attn_map.flatten(1), dim=1).mean().item()
                else: self.self_similarity = 0.0
                self.self_entropy = -(aself * torch.log(aself + 1e-10)).sum(dim=-1).mean().item()
                self.prev_self_attn_map, self.self_attn_map = aself.clone(), rearrange(aself, 'b h (ht w) d -> b h ht w d', ht=height)
            self.timestep = int(ts_val)

    joint_output = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(value.dtype)
    if encoder_hidden_states is not None:
        hidden_states, encoder_hidden_states = joint_output[:, : residual.shape[1]], joint_output[:, residual.shape[1] :]
        if not attn.context_pre_only: encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        return attn.to_out[1](attn.to_out[0](hidden_states)), encoder_hidden_states
    return attn.to_out[1](attn.to_out[0](joint_output))

def flux_attn_call2_0(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, image_rotary_emb=None, height=None, timestep=None) -> torch.FloatTensor:
    batch_size = hidden_states.shape[0] if encoder_hidden_states is None else encoder_hidden_states.shape[0]
    from .utils import flop_counter, cache_schedule, top_k_k
    # Safety check for timestep
    if timestep is not None:
        ts_val = str(int(timestep[0].item())) if torch.is_tensor(timestep) else str(int(timestep))
    else:
        ts_val = None
        
    layer_name = getattr(self, "layer_name", None)
    config = cache_schedule.get(ts_val, {}).get(layer_name, {}) if ts_val else {}
    use_cache, use_top_k = (config.get("cache", False), config.get("top_k", False)) if isinstance(config, dict) else (config, False)

    if use_cache and not use_top_k and hasattr(self, "prev_attn_output"):
        hidden_states = self.prev_attn_output
    else:
        value = attn.to_v(hidden_states)
        inner_dim, head_dim = value.shape[-1], value.shape[-1] // attn.heads
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        if encoder_hidden_states is not None:
            ev = attn.add_v_proj(encoder_hidden_states).view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value = torch.cat([ev, value], dim=2)

        if use_cache and use_top_k and hasattr(self, "prev_attn_probs"):
            attention_probs = self.prev_attn_probs
            Lq, Lk = attention_probs.shape[2], attention_probs.shape[3]
            actual_k = min(top_k_k, Lk) if top_k_k > 0 else Lk
            flop_counter.add_flops("attention_prob_v", 2 * batch_size * attn.heads * Lq * actual_k * head_dim)
            if top_k_k > 0: attention_probs = sparsify_top_k(attention_probs, top_k_k)
            hidden_states = torch.matmul(attention_probs, value)
        else:
            query, key = attn.to_q(hidden_states), attn.to_k(hidden_states)
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            if attn.norm_q is not None: query = attn.norm_q(query)
            if attn.norm_k is not None: key = attn.norm_k(key)
            if encoder_hidden_states is not None:
                eq, ek = attn.add_q_proj(encoder_hidden_states), attn.add_k_proj(encoder_hidden_states)
                eq = eq.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                ek = ek.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                if attn.norm_added_q is not None: eq = attn.norm_added_q(eq)
                if attn.norm_added_k is not None: ek = attn.norm_added_k(ek)
                query, key = torch.cat([eq, query], dim=2), torch.cat([ek, key], dim=2)
            if image_rotary_emb is not None:
                from diffusers.models.embeddings import apply_rotary_emb
                query, key = apply_rotary_emb(query, image_rotary_emb), apply_rotary_emb(key, image_rotary_emb)

            Lq, Lk = query.shape[2], key.shape[2]
            flop_counter.add_flops("attention_q_k", 2 * batch_size * attn.heads * Lq * Lk * head_dim)
            flop_counter.add_flops("attention_softmax", 3 * batch_size * attn.heads * Lq * Lk)
            actual_k = min(top_k_k, Lk) if (use_top_k and top_k_k > 0) else Lk
            flop_counter.add_flops("attention_prob_v", 2 * batch_size * attn.heads * Lq * actual_k * head_dim)
            hidden_states, attention_probs = scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
            if use_top_k and top_k_k > 0: attention_probs = sparsify_top_k(attention_probs, top_k_k)
            hidden_states = torch.matmul(attention_probs, value)
            if ts_val and layer_name:
                self.prev_attn_probs = attention_probs.clone(); self.prev_attn_output = hidden_states

        collect_cross, collect_self = getattr(self, "collect_cross_attn", False), getattr(self, "collect_self_attn", False)
        if (collect_cross or collect_self) and encoder_hidden_states is not None:
            text_length = eq.shape[2] if 'eq' in locals() else 0
            if collect_cross:
                ac = attention_probs[:, :, text_length:, :text_length].cpu()
                if hasattr(self, 'prev_attn_map'): self.similarity = F.cosine_similarity(ac.flatten(1), self.prev_attn_map.flatten(1), dim=1).mean().item()
                else: self.similarity = 0.0
                self.entropy = -(ac * torch.log(ac + 1e-10)).sum(dim=-1).mean().item()
                self.prev_attn_map, self.attn_map = ac.clone(), rearrange(ac, 'b h (ht w) d -> b h ht w d', ht=height)
            if collect_self:
                aself = attention_probs[:, :, text_length:, text_length:].cpu()
                if hasattr(self, 'prev_self_attn_map'): self.self_similarity = F.cosine_similarity(aself.flatten(1), self.prev_self_attn_map.flatten(1), dim=1).mean().item()
                else: self.self_similarity = 0.0
                self.self_entropy = -(aself * torch.log(aself + 1e-10)).sum(dim=-1).mean().item()
                self.prev_self_attn_map, self.self_attn_map = aself.clone(), rearrange(aself, 'b h (ht w) d -> b h ht w d', ht=height)
            self.timestep = int(ts_val)

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(value.dtype)
    if encoder_hidden_states is not None:
        encoder_hidden_states, hidden_states = hidden_states[:, : encoder_hidden_states.shape[1]], hidden_states[:, encoder_hidden_states.shape[1] :]
        return attn.to_out[1](attn.to_out[0](hidden_states)), attn.to_add_out(encoder_hidden_states)
    return hidden_states

def attn_call2_0(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, height=None, width=None, timestep=None, *args, **kwargs) -> torch.Tensor:
    residual = hidden_states
    if attn.spatial_norm is not None: hidden_states = attn.spatial_norm(hidden_states, temb)
    input_ndim = hidden_states.ndim
    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
    
    from .utils import flop_counter, cache_schedule, top_k_k
    ts_val, layer_name = str(int(timestep.item())), getattr(self, "layer_name", None)
    config = cache_schedule.get(ts_val, {}).get(layer_name, {})
    use_cache, use_top_k = (config.get("cache", False), config.get("top_k", False)) if isinstance(config, dict) else (config, False)

    if use_cache and not use_top_k and hasattr(self, "prev_attn_output"):
        hidden_states = self.prev_attn_output
    else:
        batch_size = hidden_states.shape[0]
        value = attn.to_v(hidden_states if encoder_hidden_states is None else encoder_hidden_states)
        inner_dim, head_dim = value.shape[-1], value.shape[-1] // attn.heads
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if use_cache and use_top_k and hasattr(self, "prev_attn_probs"):
            attention_probs = self.prev_attn_probs
            BH, Lq, Lk = attention_probs.shape[0]*attention_probs.shape[1], attention_probs.shape[2], attention_probs.shape[3]
            actual_k = min(top_k_k, Lk) if top_k_k > 0 else Lk
            flop_counter.add_flops("attention_prob_v", 2 * BH * Lq * actual_k * head_dim)
            if top_k_k > 0: attention_probs = sparsify_top_k(attention_probs, top_k_k)
            hidden_states = torch.matmul(attention_probs, value)
        else:
            query = attn.to_q(hidden_states)
            is_cross = encoder_hidden_states is not None
            if encoder_hidden_states is None: encoder_hidden_states = hidden_states
            elif attn.norm_cross: encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            key = attn.to_k(encoder_hidden_states)
            query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            BH, Lq, Lk = batch_size * attn.heads, query.shape[2], key.shape[2]
            flop_counter.add_flops("attention_q_k", 2 * BH * Lq * Lk * head_dim)
            flop_counter.add_flops("attention_softmax", 3 * BH * Lq * Lk)
            actual_k = min(top_k_k, Lk) if (use_top_k and top_k_k > 0) else Lk
            flop_counter.add_flops("attention_prob_v", 2 * BH * Lq * actual_k * head_dim)

            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, Lk, batch_size)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            hidden_states, attention_probs = scaled_dot_product_attention(query, key, value, attn_mask=attention_mask)
            if use_top_k and top_k_k > 0: attention_probs = sparsify_top_k(attention_probs, top_k_k)
            hidden_states = torch.matmul(attention_probs, value)
            if ts_val and layer_name:
                self.prev_attn_probs = attention_probs.clone(); self.prev_attn_output = hidden_states

        collect_cross, collect_self = getattr(self, "collect_cross_attn", False), getattr(self, "collect_self_attn", False)
        if (collect_cross and is_cross) or (collect_self and not is_cross):
            ac = attention_probs.cpu()
            if is_cross:
                if hasattr(self, 'prev_attn_map'): self.similarity = F.cosine_similarity(ac.flatten(1), self.prev_attn_map.flatten(1), dim=1).mean().item()
                else: self.similarity = 0.0
                self.entropy = -(ac * torch.log(ac + 1e-10)).sum(dim=-1).mean().item()
                self.prev_attn_map, self.attn_map = ac.clone(), rearrange(ac, 'b h (ht w) d -> b h ht w d', ht=height)
            else:
                if hasattr(self, 'prev_self_attn_map'): self.self_similarity = F.cosine_similarity(ac.flatten(1), self.prev_self_attn_map.flatten(1), dim=1).mean().item()
                else: self.self_similarity = 0.0
                self.self_entropy = -(ac * torch.log(ac + 1e-10)).sum(dim=-1).mean().item()
                self.prev_self_attn_map, self.self_attn_map = ac.clone(), rearrange(ac, 'b h (ht w) d -> b h ht w d', ht=height)
            self.timestep = int(ts_val)

    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim).to(value.dtype)
    hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
    if input_ndim == 4: hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
    if attn.residual_connection: hidden_states = hidden_states + residual
    return hidden_states / attn.rescale_output_factor

def attn_call(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, height=None, width=None, timestep=None, *args, **kwargs) -> torch.Tensor:
    residual = hidden_states
    if attn.spatial_norm is not None: hidden_states = attn.spatial_norm(hidden_states, temb)
    input_ndim = hidden_states.ndim
    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
    
    from .utils import flop_counter, cache_schedule, top_k_k
    # Safety check for timestep
    if timestep is not None:
        ts_val = str(int(timestep.item())) if torch.is_tensor(timestep) else str(int(timestep))
    else:
        ts_val = None
        
    layer_name = getattr(self, "layer_name", None)
    config = cache_schedule.get(ts_val, {}).get(layer_name, {}) if ts_val else {}
    use_cache, use_top_k = (config.get("cache", False), config.get("top_k", False)) if isinstance(config, dict) else (config, False)

    if use_cache and not use_top_k and hasattr(self, "prev_attn_output"):
        hidden_states = self.prev_attn_output
    else:
        value = attn.to_v(hidden_states if encoder_hidden_states is None else encoder_hidden_states)
        value = attn.head_to_batch_dim(value)

        if use_cache and use_top_k and hasattr(self, "prev_attn_probs"):
            attention_probs = self.prev_attn_probs
            BH, Lq, Lk = attention_probs.shape[0], attention_probs.shape[1], attention_probs.shape[2]
            actual_k = min(top_k_k, Lk) if top_k_k > 0 else Lk
            flop_counter.add_flops("attention_prob_v", 2 * BH * Lq * actual_k * value.shape[2])
            if top_k_k > 0: attention_probs = sparsify_top_k(attention_probs, top_k_k)
            hidden_states = torch.bmm(attention_probs, value)
        else:
            query = attn.to_q(hidden_states)
            is_cross = encoder_hidden_states is not None
            if encoder_hidden_states is None: encoder_hidden_states = hidden_states
            elif attn.norm_cross: encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            key = attn.to_k(encoder_hidden_states)
            query, key = attn.head_to_batch_dim(query), attn.head_to_batch_dim(key)

            BH, Lq, Lk = query.shape[0], query.shape[1], key.shape[1]
            flop_counter.add_flops("attention_q_k", 2 * BH * Lq * Lk * query.shape[2])
            flop_counter.add_flops("attention_softmax", 3 * BH * Lq * Lk)
            actual_k = min(top_k_k, Lk) if (use_top_k and top_k_k > 0) else Lk
            flop_counter.add_flops("attention_prob_v", 2 * BH * Lq * actual_k * value.shape[2])

            attention_mask = attn.prepare_attention_mask(attention_mask, Lk, hidden_states.shape[0])
            attention_probs = attn.get_attention_scores(query, key, attention_mask)
            if use_top_k and top_k_k > 0: attention_probs = sparsify_top_k(attention_probs, top_k_k)
            
            collect_cross, collect_self = getattr(self, "collect_cross_attn", False), getattr(self, "collect_self_attn", False)
            if (collect_cross and is_cross) or (collect_self and not is_cross):
                ac = attention_probs.cpu()
                if is_cross:
                    if hasattr(self, 'prev_attn_map'): self.similarity = F.cosine_similarity(ac.flatten(1), self.prev_attn_map.flatten(1), dim=1).mean().item()
                    else: self.similarity = 0.0
                    self.entropy = -(ac * torch.log(ac + 1e-10)).sum(dim=-1).mean().item()
                    self.prev_attn_map, self.attn_map = ac.clone(), rearrange(ac, 'b (ht w) d -> b d ht w', ht=height)
                else:
                    if hasattr(self, 'prev_self_attn_map'): self.self_similarity = F.cosine_similarity(ac.flatten(1), self.prev_self_attn_map.flatten(1), dim=1).mean().item()
                    else: self.self_similarity = 0.0
                    self.self_entropy = -(ac * torch.log(ac + 1e-10)).sum(dim=-1).mean().item()
                    self.prev_self_attn_map, self.self_attn_map = ac.clone(), rearrange(ac, 'b (ht w) d -> b d ht w', ht=height)
                self.timestep = int(ts_val)
            
            hidden_states = torch.bmm(attention_probs, value)
            if ts_val and layer_name:
                self.prev_attn_probs = attention_probs.clone(); self.prev_attn_output = hidden_states

    hidden_states = attn.to_out[1](attn.to_out[0](attn.batch_to_head_dim(hidden_states)))
    if input_ndim == 4: hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
    if attn.residual_connection: hidden_states = hidden_states + residual
    return hidden_states / attn.rescale_output_factor

def lora_attn_call(self, attn: Attention, hidden_states, height, width, *args, **kwargs):
    attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
    attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
    attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
    attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)
    attn._modules.pop("processor")
    attn.processor = AttnProcessor()
    attn.processor.__call__ = attn_call.__get__(attn.processor, AttnProcessor)
    return attn.processor(attn, hidden_states, height, width, *args, **kwargs)

def lora_attn_call2_0(self, attn: Attention, hidden_states, height, width, *args, **kwargs):
    attn.to_q.lora_layer = self.to_q_lora.to(hidden_states.device)
    attn.to_k.lora_layer = self.to_k_lora.to(hidden_states.device)
    attn.to_v.lora_layer = self.to_v_lora.to(hidden_states.device)
    attn.to_out[0].lora_layer = self.to_out_lora.to(hidden_states.device)
    attn._modules.pop("processor")
    attn.processor = AttnProcessor2_0()
    attn.processor.__call__ = attn_call2_0.__get__(attn.processor, AttnProcessor2_0)
    return attn.processor(attn, hidden_states, height, width, *args, **kwargs)

def UNet2DConditionModelForward(self, sample, timestep, encoder_hidden_states, class_labels=None, timestep_cond=None, attention_mask=None, cross_attention_kwargs=None, added_cond_kwargs=None, down_block_additional_residuals=None, mid_block_additional_residual=None, down_intrablock_additional_residuals=None, encoder_attention_mask=None, return_dict=True):
    default_overall_up_factor = 2**self.num_upsamplers
    forward_upsample_size = False
    for dim in sample.shape[-2:]:
        if dim % default_overall_up_factor != 0: forward_upsample_size = True; break
    if attention_mask is not None: attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0; attention_mask = attention_mask.unsqueeze(1)
    if encoder_attention_mask is not None: encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0; encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
    if self.config.center_input_sample: sample = 2 * sample - 1.0
    t_emb = self.get_time_embed(sample=sample, timestep=timestep)
    emb = self.time_embedding(t_emb, timestep_cond)
    class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
    if class_emb is not None: emb = torch.cat([emb, class_emb], dim=-1) if self.config.class_embeddings_concat else emb + class_emb
    aug_emb = self.get_aug_embed(emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)
    if self.config.addition_embed_type == "image_hint": aug_emb, hint = aug_emb; sample = torch.cat([sample, hint], dim=1)
    emb = emb + aug_emb if aug_emb is not None else emb
    if self.time_embed_act is not None: emb = self.time_embed_act(emb)
    encoder_hidden_states = self.process_encoder_hidden_states(encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)
    sample = self.conv_in(sample)
    if cross_attention_kwargs is None: cross_attention_kwargs = {'timestep' : timestep}
    else: cross_attention_kwargs['timestep'] = timestep
    cross_attention_kwargs = cross_attention_kwargs.copy(); lora_scale = cross_attention_kwargs.pop("scale", 1.0)
    if USE_PEFT_BACKEND: scale_lora_layers(self, lora_scale)
    down_block_res_samples = (sample,)
    for db in self.down_blocks:
        if hasattr(db, "has_cross_attention") and db.has_cross_attention: sample, res = db(sample, emb, encoder_hidden_states, attention_mask, cross_attention_kwargs, encoder_attention_mask)
        else: sample, res = db(sample, emb)
        down_block_res_samples += res
    if mid_block_additional_residual is not None: sample = sample + mid_block_additional_residual
    if self.mid_block is not None:
        if hasattr(self.mid_block, "has_cross_attention") and self.mid_block.has_cross_attention: sample = self.mid_block(sample, emb, encoder_hidden_states, attention_mask, cross_attention_kwargs, encoder_attention_mask)
        else: sample = self.mid_block(sample, emb)
    for i, ub in enumerate(self.up_blocks):
        res = down_block_res_samples[-len(ub.resnets):]; down_block_res_samples = down_block_res_samples[:-len(ub.resnets)]
        upsample_size = down_block_res_samples[-1].shape[2:] if (i != len(self.up_blocks)-1 and forward_upsample_size) else None
        if hasattr(ub, "has_cross_attention") and ub.has_cross_attention: sample = ub(sample, emb, res, encoder_hidden_states, cross_attention_kwargs, upsample_size, attention_mask, encoder_attention_mask)
        else: sample = ub(sample, emb, res, upsample_size)
    if self.conv_norm_out: sample = self.conv_act(self.conv_norm_out(sample))
    sample = self.conv_out(sample)
    if USE_PEFT_BACKEND: unscale_lora_layers(self, lora_scale)
    return UNet2DConditionOutput(sample=sample) if return_dict else (sample,)

def SD3Transformer2DModelForward(self, hidden_states, encoder_hidden_states=None, pooled_projections=None, timestep=None, block_controlnet_hidden_states=None, joint_attention_kwargs=None, return_dict=True):
    lora_scale = joint_attention_kwargs.pop("scale", 1.0) if joint_attention_kwargs else 1.0
    if USE_PEFT_BACKEND: scale_lora_layers(self, lora_scale)
    height, width = hidden_states.shape[-2:]
    hidden_states = self.pos_embed(hidden_states); temb = self.time_text_embed(timestep, pooled_projections); encoder_hidden_states = self.context_embedder(encoder_hidden_states)
    for i, block in enumerate(self.transformer_blocks):
        encoder_hidden_states, hidden_states = block(hidden_states, encoder_hidden_states, temb, timestep=timestep, height=height // self.config.patch_size)
    hidden_states = self.norm_out(hidden_states, temb); hidden_states = self.proj_out(hidden_states)
    p = self.config.patch_size; h, w = height // p, width // p
    hidden_states = hidden_states.reshape(hidden_states.shape[0], h, w, p, p, self.out_channels)
    output = torch.einsum("nhwpqc->nchpwq", hidden_states).reshape(hidden_states.shape[0], self.out_channels, height, width)
    if USE_PEFT_BACKEND: unscale_lora_layers(self, lora_scale)
    return Transformer2DModelOutput(sample=output) if return_dict else (output,)

def FluxTransformer2DModelForward(self, hidden_states, encoder_hidden_states=None, pooled_projections=None, timestep=None, img_ids=None, txt_ids=None, guidance=None, joint_attention_kwargs=None, controlnet_block_samples=None, controlnet_single_block_samples=None, return_dict=True, controlnet_blocks_repeat=False, height=None, width=None):
    lora_scale = joint_attention_kwargs.pop("scale", 1.0) if joint_attention_kwargs else 1.0
    if USE_PEFT_BACKEND: scale_lora_layers(self, lora_scale)
    hidden_states = self.x_embedder(hidden_states)
    timestep = timestep.to(hidden_states.dtype) * 1000
    temb = self.time_text_embed(timestep, guidance.to(hidden_states.dtype)*1000, pooled_projections) if guidance is not None else self.time_text_embed(timestep, pooled_projections)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)
    ids = torch.cat((txt_ids[0] if txt_ids.ndim==3 else txt_ids, img_ids[0] if img_ids.ndim==3 else img_ids), dim=0)
    rotary = self.pos_embed(ids)
    for block in self.transformer_blocks: encoder_hidden_states, hidden_states = block(hidden_states, encoder_hidden_states, temb, rotary, joint_attention_kwargs, timestep=timestep, height=height // self.config.patch_size)
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
    for block in self.single_transformer_blocks: hidden_states = block(hidden_states, temb, rotary, joint_attention_kwargs)
    hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, :]
    output = self.proj_out(self.norm_out(hidden_states, temb))
    if USE_PEFT_BACKEND: unscale_lora_layers(self, lora_scale)
    return Transformer2DModelOutput(sample=output) if return_dict else (output,)

def Transformer2DModelForward(self, hidden_states, encoder_hidden_states=None, timestep=None, added_cond_kwargs=None, class_labels=None, cross_attention_kwargs=None, attention_mask=None, encoder_attention_mask=None, return_dict=True):
    if attention_mask is not None and attention_mask.ndim == 2: attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0; attention_mask = attention_mask.unsqueeze(1)
    if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2: encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0; encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
    if self.is_input_continuous:
        batch_size, _, height, width = hidden_states.shape; residual = hidden_states; hidden_states, inner_dim = self._operate_on_continuous_inputs(hidden_states)
    elif self.is_input_patches:
        height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
        hidden_states, encoder_hidden_states, timestep, _ = self._operate_on_patched_inputs(hidden_states, encoder_hidden_states, timestep, added_cond_kwargs)
    cross_attention_kwargs['height'], cross_attention_kwargs['width'] = height, width
    for block in self.transformer_blocks: hidden_states = block(hidden_states, attention_mask, encoder_hidden_states, encoder_attention_mask, timestep, cross_attention_kwargs, class_labels)
    if self.is_input_continuous: output = self._get_output_for_continuous_inputs(hidden_states, residual, batch_size, height, width, inner_dim)
    elif self.is_input_patches: output = self._get_output_for_patched_inputs(hidden_states, timestep, class_labels, None, height, width)
    return Transformer2DModelOutput(sample=output) if return_dict else (output,)

def BasicTransformerBlockForward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, timestep=None, cross_attention_kwargs=None, class_labels=None, added_cond_kwargs=None):
    batch_size = hidden_states.shape[0]
    if self.norm_type == "ada_norm": norm = self.norm1(hidden_states, timestep)
    elif self.norm_type == "ada_norm_zero": norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype)
    else: norm = self.norm1(hidden_states)
    if self.pos_embed is not None: norm = self.pos_embed(norm)
    cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs else {}
    attn_params = set(inspect.signature(self.attn1.processor.__call__).parameters.keys())
    hidden_states = (gate_msa.unsqueeze(1) * self.attn1(norm, encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None, attention_mask=attention_mask, **{k:v for k,v in cross_attention_kwargs.items() if k in attn_params})) + hidden_states
    if self.attn2 is not None:
        norm2 = self.norm2(hidden_states, timestep) if self.norm_type == "ada_norm" else self.norm2(hidden_states)
        hidden_states = self.attn2(norm2, encoder_hidden_states=encoder_hidden_states, attention_mask=encoder_attention_mask, **cross_attention_kwargs) + hidden_states
    ff_norm = self.norm3(hidden_states)
    if self.norm_type == "ada_norm_zero": ff_norm = ff_norm * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
    hidden_states = (gate_mlp.unsqueeze(1) * self.ff(ff_norm)) + hidden_states
    return hidden_states.squeeze(1) if hidden_states.ndim == 4 else hidden_states

def JointTransformerBlockForward(self, hidden_states, encoder_hidden_states, temb, height=None, timestep=None):
    if self.use_dual_attention:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = self.norm1(
            hidden_states, emb=temb
        )
    else:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

    if self.context_pre_only:
        norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
    else:
        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            encoder_hidden_states, emb=temb
        )
    
    ts_val = str(int(timestep[0].item())) if timestep is not None else None
    layer_name = getattr(self, "layer_name", None)
    from .utils import cache_schedule
    config = cache_schedule.get(ts_val, {}).get(layer_name, {})
    use_cache = config if isinstance(config, bool) else config.get("cache", False)
    
    if use_cache and hasattr(self, "prev_attn_output"):
        attn_output, context_attn_output = self.prev_attn_output, self.prev_context_attn_output
    else:
        attn_output, context_attn_output = self.attn(norm_hidden_states, encoder_hidden_states=norm_encoder_hidden_states, timestep=timestep, height=height)
        if ts_val and layer_name: self.prev_attn_output, self.prev_context_attn_output = attn_output, context_attn_output

    hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
    if self.use_dual_attention:
        attn_output2 = self.attn2(hidden_states=norm_hidden_states2, timestep=timestep, height=height)
        hidden_states = hidden_states + gate_msa2.unsqueeze(1) * attn_output2

    hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.ff(self.norm2(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None])
    if not self.context_pre_only:
        encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * self.ff_context(self.norm2_context(encoder_hidden_states) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None])
    return encoder_hidden_states, hidden_states

def FluxTransformerBlockForward(self, hidden_states, encoder_hidden_states, temb, image_rotary_emb=None, joint_attention_kwargs=None, height=None, width=None, timestep=None):
    norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
    norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)
    
    joint_attention_kwargs = joint_attention_kwargs or {}
    
    # Logic for caching
    ts_val = str(int(timestep[0].item())) if timestep is not None else None
    layer_name = getattr(self, "layer_name", None)
    from .utils import cache_schedule
    config = cache_schedule.get(ts_val, {}).get(layer_name, {})
    use_cache = config if isinstance(config, bool) else config.get("cache", False)

    if use_cache and hasattr(self, "prev_attn_output"):
        attn_output, context_attn_output = self.prev_attn_output, self.prev_context_attn_output
    else:
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            timestep=timestep,
            height=height,
            **joint_attention_kwargs
        )
        if ts_val and layer_name: self.prev_attn_output, self.prev_context_attn_output = attn_output, context_attn_output

    hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
    hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.ff(self.norm2(hidden_states) * (1 + scale_mlp[:, None]) + shift_mlp[:, None])
    encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
    encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * self.ff_context(self.norm2_context(encoder_hidden_states) * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None])
    return encoder_hidden_states, hidden_states
