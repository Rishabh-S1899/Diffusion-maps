import torch
from diffusers import StableDiffusion3Pipeline
from attention_map_diffusers import init_pipeline
from attention_map_diffusers.utils import register_flops_hook, flop_counter
import os

def test_flops_real():
    print("Loading SD3.5 for FLOPs profiling...")
    try:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium", 
            torch_dtype=torch.bfloat16
        )
        pipe = pipe.to("cuda")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Initialize pipeline with our custom processors
    pipe = init_pipeline(pipe, collect_cross_attn=False, collect_self_attn=False)

    # Register FLOP hooks on the transformer
    print("Registering FLOP hooks...")
    pipe.transformer = register_flops_hook(pipe.transformer)

    prompt = "a futuristic city"
    
    # We only need 1 step to measure per-step FLOPs
    print("Running 1 inference step...")
    flop_counter.reset()
    
    with torch.no_grad():
        pipe(prompt, num_inference_steps=1)

    print("--- PROFILING RESULTS ---")
    flop_counter.print_summary()
    
    # Breakdown of top contributors
    sorted_layers = sorted(flop_counter.layer_flops.items(), key=lambda x: x[1], reverse=True)
    print("Top 5 FLOP contributors:")
    for name, flops in sorted_layers[:5]:
        print(f"  {name}: {flops / 1e9:.2f} GFLOPs")

if __name__ == "__main__":
    test_flops_real()
