import os

import torch
import torch.nn.functional as F
from torchvision.transforms import ToPILImage

from diffusers.models.attention_processor import (
    AttnProcessor,
    AttnProcessor2_0,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    JointAttnProcessor2_0,
    FluxAttnProcessor2_0
)

from .modules import *


def hook_function(name, collect_cross=True, collect_self=True, detach=True):
    def forward_hook(module, input, output):
        has_cross = hasattr(module.processor, "attn_map")
        has_self = hasattr(module.processor, "self_attn_map")
        
        if (collect_cross and has_cross) or (collect_self and has_self):
            timestep = module.processor.timestep
            
            attn_maps[timestep] = attn_maps.get(timestep, dict())
            attn_maps[timestep][name] = {}
            
            if collect_cross and has_cross:
                attn_maps[timestep][name].update({
                    'map': module.processor.attn_map.cpu() if detach else module.processor.attn_map,
                    'similarity': getattr(module.processor, 'similarity', 0.0),
                    'entropy': getattr(module.processor, 'entropy', 0.0)
                })
                del module.processor.attn_map
                
            if collect_self and has_self:
                attn_maps[timestep][name].update({
                    'self_map': module.processor.self_attn_map.cpu() if detach else module.processor.self_attn_map,
                    'self_similarity': getattr(module.processor, 'self_similarity', 0.0),
                    'self_entropy': getattr(module.processor, 'self_entropy', 0.0)
                })
                del module.processor.self_attn_map
    
    return forward_hook

def save_attention_stats(attn_maps, base_dir='attn_stats'):
    """Save only similarity and entropy statistics, not the full maps"""
    import json
    
    os.makedirs(base_dir, exist_ok=True)
    
    stats = {}
    for timestep, layers in attn_maps.items():
        stats[timestep] = {}
        for layer, data in layers.items():
            stats[timestep][layer] = {}
            if 'similarity' in data:
                stats[timestep][layer].update({
                    'similarity': data['similarity'],
                    'entropy': data['entropy']
                })
            if 'self_similarity' in data:
                stats[timestep][layer].update({
                    'self_similarity': data['self_similarity'],
                    'self_entropy': data['self_entropy']
                })
    
    with open(os.path.join(base_dir, 'statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    # print(f"Statistics saved to {base_dir}/statistics.json")

def register_cross_attention_hook(model, hook_function, target_name, collect_cross=True, collect_self=True):
    for name, module in model.named_modules():
        if not name.endswith(target_name):
            continue

        module.processor.collect_cross_attn = collect_cross
        module.processor.collect_self_attn = collect_self

        hook = module.register_forward_hook(hook_function(name, collect_cross=collect_cross, collect_self=collect_self))
    
    return model


def replace_call_method_for_unet(model):
    if model.__class__.__name__ == 'UNet2DConditionModel':
        from diffusers.models.unets import UNet2DConditionModel
        model.forward = UNet2DConditionModelForward.__get__(model, UNet2DConditionModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'Transformer2DModel':
            from diffusers.models import Transformer2DModel
            layer.forward = Transformer2DModelForward.__get__(layer, Transformer2DModel)
        
        elif layer.__class__.__name__ == 'BasicTransformerBlock':
            from diffusers.models.attention import BasicTransformerBlock
            layer.forward = BasicTransformerBlockForward.__get__(layer, BasicTransformerBlock)
        
        replace_call_method_for_unet(layer)
    
    return model


# TODO: implement
# def replace_call_method_for_sana(model):
#     if model.__class__.__name__ == 'SanaTransformer2DModel':
#         from diffusers.models.transformers import SanaTransformer2DModel
#         model.forward = SanaTransformer2DModelForward.__get__(model, SanaTransformer2DModel)

#     for name, layer in model.named_children():
        
#         if layer.__class__.__name__ == 'SanaTransformerBlock':
#             from diffusers.models.transformers.sana_transformer import SanaTransformerBlock
#             layer.forward = SanaTransformerBlockForward.__get__(layer, SanaTransformerBlock)
        
#         replace_call_method_for_sana(layer)
    
#     return model


def replace_call_method_for_sd3(model):
    if model.__class__.__name__ == 'SD3Transformer2DModel':
        from diffusers.models.transformers import SD3Transformer2DModel
        model.forward = SD3Transformer2DModelForward.__get__(model, SD3Transformer2DModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'JointTransformerBlock':
            from diffusers.models.attention import JointTransformerBlock
            layer.forward = JointTransformerBlockForward.__get__(layer, JointTransformerBlock)
        
        replace_call_method_for_sd3(layer)
    
    return model


def replace_call_method_for_flux(model):
    if model.__class__.__name__ == 'FluxTransformer2DModel':
        from diffusers.models.transformers import FluxTransformer2DModel
        model.forward = FluxTransformer2DModelForward.__get__(model, FluxTransformer2DModel)

    for name, layer in model.named_children():
        
        if layer.__class__.__name__ == 'FluxTransformerBlock':
            from diffusers.models.transformers.transformer_flux import FluxTransformerBlock
            layer.forward = FluxTransformerBlockForward.__get__(layer, FluxTransformerBlock)
        
        replace_call_method_for_flux(layer)
    
    return model


def init_pipeline(pipeline, collect_cross_attn=True, collect_self_attn=True):
    AttnProcessor.__call__ = attn_call
    AttnProcessor2_0.__call__ = attn_call2_0
    LoRAAttnProcessor.__call__ = lora_attn_call
    LoRAAttnProcessor2_0.__call__ = lora_attn_call2_0
    if 'transformer' in vars(pipeline).keys():
        if pipeline.transformer.__class__.__name__ == 'SD3Transformer2DModel':
            JointAttnProcessor2_0.__call__ = joint_attn_call2_0
            pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn',
                                                               collect_cross=collect_cross_attn,
                                                               collect_self=collect_self_attn)
            pipeline.transformer = replace_call_method_for_sd3(pipeline.transformer)
        
        elif pipeline.transformer.__class__.__name__ == 'FluxTransformer2DModel':
            from diffusers import FluxPipeline
            FluxAttnProcessor2_0.__call__ = flux_attn_call2_0
            FluxPipeline.__call__ = FluxPipeline_call
            pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn',
                                                               collect_cross=collect_cross_attn,
                                                               collect_self=collect_self_attn)
            pipeline.transformer = replace_call_method_for_flux(pipeline.transformer)

        # TODO: implement
        # elif pipeline.transformer.__class__.__name__ == 'SanaTransformer2DModel':
        #     from diffusers import SanaPipeline
        #     SanaPipeline.__call__ == SanaPipeline_call
        #     pipeline.transformer = register_cross_attention_hook(pipeline.transformer, hook_function, 'attn2')
        #     pipeline.transformer = replace_call_method_for_sana(pipeline.transformer)

    else:
        if pipeline.unet.__class__.__name__ == 'UNet2DConditionModel':
            pipeline.unet = register_cross_attention_hook(pipeline.unet, hook_function, 'attn1',
                                                         collect_cross=False, collect_self=collect_self_attn)
            pipeline.unet = register_cross_attention_hook(pipeline.unet, hook_function, 'attn2',
                                                         collect_cross=collect_cross_attn, collect_self=False)
            pipeline.unet = replace_call_method_for_unet(pipeline.unet)


    return pipeline


def process_token(token, startofword):
    if '</w>' in token:
        token = token.replace('</w>', '')
        if startofword:
            token = '<' + token + '>'
        else:
            token = '-' + token + '>'
            startofword = True
    elif token not in ['<|startoftext|>', '<|endoftext|>']:
        if startofword:
            token = '<' + token + '-'
            startofword = False
        else:
            token = '-' + token + '-'
    return token, startofword


def save_attention_image(attn_map, tokens, batch_dir, to_pil):
    startofword = True
    for i, (token, a) in enumerate(zip(tokens, attn_map[:len(tokens)])):
        token, startofword = process_token(token, startofword)
        to_pil(a.to(torch.float32)).save(os.path.join(batch_dir, f'{i}-{token}.png'))


def save_attention_maps(attn_maps, tokenizer, prompts, base_dir='attn_maps', unconditional=True):
    to_pil = ToPILImage()
    
    token_ids = tokenizer(prompts)['input_ids']
    token_ids = token_ids if token_ids and isinstance(token_ids[0], list) else [token_ids]
    total_tokens = [tokenizer.convert_ids_to_tokens(token_id) for token_id in token_ids]
    
    os.makedirs(base_dir, exist_ok=True)
    
    total_attn_map = list(list(attn_maps.values())[0].values())[0].sum(1)
    if unconditional:
        total_attn_map = total_attn_map.chunk(2)[1]  # (batch, height, width, attn_dim)
    total_attn_map = total_attn_map.permute(0, 3, 1, 2)
    total_attn_map = torch.zeros_like(total_attn_map)
    total_attn_map_shape = total_attn_map.shape[-2:]
    total_attn_map_number = 0
    
    for timestep, layers in attn_maps.items():
        timestep_dir = os.path.join(base_dir, f'{timestep}')
        os.makedirs(timestep_dir, exist_ok=True)
        
        for layer, attn_map in layers.items():
            layer_dir = os.path.join(timestep_dir, f'{layer}')
            os.makedirs(layer_dir, exist_ok=True)
            
            attn_map = attn_map.sum(1).squeeze(1).permute(0, 3, 1, 2)
            if unconditional:
                attn_map = attn_map.chunk(2)[1]
            
            resized_attn_map = F.interpolate(attn_map, size=total_attn_map_shape, mode='bilinear', align_corners=False)
            total_attn_map += resized_attn_map
            total_attn_map_number += 1
            
            for batch, (tokens, attn) in enumerate(zip(total_tokens, attn_map)):
                batch_dir = os.path.join(layer_dir, f'batch-{batch}')
                os.makedirs(batch_dir, exist_ok=True)
                save_attention_image(attn, tokens, batch_dir, to_pil)
    
    total_attn_map /= total_attn_map_number
    for batch, (attn_map, tokens) in enumerate(zip(total_attn_map, total_tokens)):
        batch_dir = os.path.join(base_dir, f'batch-{batch}')
        os.makedirs(batch_dir, exist_ok=True)
        save_attention_image(attn_map, tokens, batch_dir, to_pil)