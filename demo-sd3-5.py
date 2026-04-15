import torch
import os
import random
import re
import argparse
from datasets import load_dataset
from diffusers import StableDiffusion3Pipeline

from attention_map_diffusers import init_pipeline
from attention_map_diffusers.utils import register_flops_hook, flop_counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=int, default=1, help="Number of prompts to process")
    parser.add_argument("--profile", action="store_true", help="Profile FLOPs")
    parser.add_argument("--cache_schedule", type=str, default=None, help="Path to cache schedule JSON")
    parser.add_argument("--quantize_cache", action="store_true", help="Quantize cached hidden states to 8-bit")
    parser.add_argument("--output_dir", type=str, default="outputs_inference", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    # 1. Load prompts
    random.seed(args.seed)
    ds = load_dataset("nateraw/parti-prompts", split="train")
    all_prompts = sorted(ds['Prompt']) 
    random_prompts = random.sample(all_prompts, args.prompts)

    # 2. Setup Pipeline
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    # Disable stats collection for maximum speed
    pipe = init_pipeline(pipe, collect_cross_attn=False, collect_self_attn=False, 
                         cache_schedule_path=args.cache_schedule, quantize_cache=args.quantize_cache)

    if args.profile:
        pipe.transformer = register_flops_hook(pipe.transformer)

    # 3. Process and Save
    for i, prompt in enumerate(random_prompts):
        print(f"Inference for prompt {i+1}/{args.prompts}")
        clean_name = re.sub(r'[^\w\s]', '', prompt[:20]).strip().replace(' ', '_')
        prompt_dir = os.path.join(args.output_dir, clean_name)
        os.makedirs(prompt_dir, exist_ok=True)
        
        if args.profile: flop_counter.reset()
        
        # --- UPDATE 1: Clear BOTH Attention and MLP cache states ---
        for _, module in pipe.transformer.named_modules():
            if hasattr(module, 'prev_attn_output'): delattr(module, 'prev_attn_output')
            if hasattr(module, 'prev_context_attn_output'): delattr(module, 'prev_context_attn_output')
            if hasattr(module, 'prev_mlp_output'): delattr(module, 'prev_mlp_output')
            if hasattr(module, 'prev_context_mlp_output'): delattr(module, 'prev_context_mlp_output')

        # --- UPDATE 2: Pre-compute Text Embeddings to isolate VRAM ---
        with torch.no_grad():
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                prompt=prompt, prompt_2=None, prompt_3=None
            )

        # Reset peak memory AFTER text encoding, BEFORE the transformer starts
        if torch.cuda.is_available(): 
            torch.cuda.reset_peak_memory_stats()
        
        # Run inference using the pre-computed embeddings
        with torch.no_grad():
            image = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                num_inference_steps=15, 
                guidance_scale=4.5
            ).images[0]
        
        if args.profile:
            print(f"\nFLOPs Summary for: {prompt[:50]}...")
            flop_counter.print_summary()
        
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"Transformer Peak VRAM Usage: {peak_vram:.4f} GB")

        image.save(os.path.join(prompt_dir, "result.png"))
        with open(os.path.join(prompt_dir, "prompt.txt"), 'w') as f: f.write(prompt)

if __name__ == "__main__":
    main()