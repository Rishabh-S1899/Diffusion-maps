import torch
import os
import random
import re
import argparse
from datasets import load_dataset
from diffusers import StableDiffusion3Pipeline

from attention_map_diffusers import attn_maps, init_pipeline
from attention_map_diffusers.utils import save_attention_stats 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect_self", action="store_true", default=False, help="Collect self-attention stats")
    parser.add_argument("--prompts", type=int, default=500, help="Number of prompts to process")
    parser.add_argument("--output_dir", type=str, default="outputs_stats", help="Base output directory")
    args = parser.parse_args()

    # 1. Load prompts
    ds = load_dataset("nateraw/parti-prompts", split="train")
    random_prompts = random.sample(ds['Prompt'], args.prompts)

    # 2. Setup Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    # Enable stats collection (Cross is enabled if collect_cross_attn=True)
    pipe = init_pipeline(pipe, collect_cross_attn=True, collect_self_attn=args.collect_self)

    # 3. Process and Save
    for i, prompt in enumerate(random_prompts):
        print(f"Collecting stats for prompt {i+1}/{args.prompts}")
        clean_name = re.sub(r'[^\w\s]', '', prompt[:20]).strip().replace(' ', '_')
        prompt_dir = os.path.join(args.output_dir, clean_name)
        os.makedirs(prompt_dir, exist_ok=True)

        # Clear prev state for stats collection
        for _, module in pipe.transformer.named_modules():
            if hasattr(module, 'processor'):
                if hasattr(module.processor, 'prev_attn_map'): delattr(module.processor, 'prev_attn_map')
                if hasattr(module.processor, 'prev_self_attn_map'): delattr(module.processor, 'prev_self_attn_map')

        # Run inference
        with torch.no_grad():
            pipe(prompt, num_inference_steps=15, guidance_scale=4.5)
        
        # Save results (only statistics.json)
        save_attention_stats(attn_maps, base_dir=prompt_dir)
        with open(os.path.join(prompt_dir, "prompt.txt"), 'w') as f: f.write(prompt)
        
        # Clear maps for next prompt
        attn_maps.clear()

if __name__ == "__main__":
    main()
