import torch
import os
import sys
import random
import re
import argparse
from datasets import load_dataset
from diffusers import StableDiffusion3Pipeline

from attention_map_diffusers import attn_maps, init_pipeline
from attention_map_diffusers.utils import save_attention_stats 

# 0. Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--collect_cross", action="store_true", default=False, help="Collect cross-attention stats")
parser.add_argument("--no_cross", action="store_false", dest="collect_cross", help="Disable cross-attention stats")
parser.add_argument("--collect_self", action="store_true", default=False, help="Collect self-attention stats")
parser.add_argument("--prompts", type=int, default=500, help="Number of prompts to process")
parser.add_argument("--profile", action="store_true", help="Profile FLOPs")
parser.add_argument("--cache_schedule", type=str, default=None, help="Path to cache schedule JSON")
args = parser.parse_args()

# 1. Load and sample prompts
ds = load_dataset("nateraw/parti-prompts", split="train")
random_prompts = random.sample(ds['Prompt'], args.prompts)

# 2. Setup Pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")
pipe = init_pipeline(pipe, collect_cross_attn=args.collect_cross, collect_self_attn=args.collect_self, 
                     cache_schedule_path=args.cache_schedule)

if args.profile:
    from attention_map_diffusers.utils import register_flops_hook, flop_counter
    pipe.transformer = register_flops_hook(pipe.transformer)

# 3. Process and Save
for i, prompt in enumerate(random_prompts):
    print(f"Processing prompt {i+1}/{args.prompts}")
    clean_prompt_snippet = re.sub(r'[^\w\s]', '', prompt[:20]).strip().replace(' ', '_')
    output_dir = f"outputs_parti-prompts/{clean_prompt_snippet}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    if args.profile:
        flop_counter.reset()
    
    ###############################################################
    # CRITICAL: Clear prev_attn_map and cache before each new prompt
    for name, module in pipe.transformer.named_modules():
        # Clear attention stats cache
        if hasattr(module, 'processor'):
            if hasattr(module.processor, 'prev_attn_map'):
                delattr(module.processor, 'prev_attn_map')
            if hasattr(module.processor, 'prev_self_attn_map'):
                delattr(module.processor, 'prev_self_attn_map')
        
        # Clear attention output cache (on the blocks)
        if hasattr(module, 'prev_attn_output'):
            delattr(module, 'prev_attn_output')
        if hasattr(module, 'prev_context_attn_output'):
            delattr(module, 'prev_context_attn_output')
    ###############################################################
    # Run inference
    image = pipe(prompt, num_inference_steps=15, guidance_scale=4.5).images[0]

    if args.profile:
        print(f"\nFLOPs for prompt: {prompt[:50]}...")
        flop_counter.print_summary()
    
    # Save image
    image.save(os.path.join(output_dir, "result.png"))
    with open(os.path.join(output_dir, "prompt.txt"), 'w') as f:
        f.write(prompt)

    save_attention_stats(attn_maps, base_dir=output_dir)
    
    # Clear for next prompt
    attn_maps.clear()

print("Processing complete with statistics saved for each prompt.")