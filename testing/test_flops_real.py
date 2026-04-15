import torch
from diffusers import StableDiffusion3Pipeline
from attention_map_diffusers import init_pipeline
from attention_map_diffusers.utils import register_flops_hook, flop_counter
import os
import json

def test_flops_comparison():
    print("Loading SD3.5 for FLOPs comparison...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", 
        torch_dtype=torch.bfloat16
    ).to("cuda")

    # 1. Baseline Run (init_pipeline called, but cache empty/disabled)
    print("\n--- BASELINE RUN (No Caching) ---")
    pipe = init_pipeline(pipe, collect_cross_attn=False, collect_self_attn=False)
    pipe.transformer = register_flops_hook(pipe.transformer)
    
    flop_counter.reset()
    with torch.no_grad():
        pipe("a cat", num_inference_steps=15,guidance_scale=4.5) # 2 steps to allow caching to trigger
    
    baseline_flops = flop_counter.total_flops
    flop_counter.print_summary()

    # 2. Cached Run
    print("\n--- CACHED RUN (50% Layers Skipped) ---")
    # Generate a dummy schedule for 2nd step
    # Timesteps for 2 steps are usually [1000, 500] or similar
    # schedule = {"500": {f"transformer_blocks.{i}": True for i in range(0, 38, 2)}}
    # with open("temp_test_schedule.json", "w") as f:
    #     json.dump(schedule, f)

    # Re-init with schedule
    pipe = init_pipeline(pipe, collect_cross_attn=False, collect_self_attn=False, 
                         cache_schedule_path="sample_cache_schedule.json")
    
    # Reset model state
    for _, module in pipe.transformer.named_modules():
        if hasattr(module, 'prev_attn_output'): delattr(module, 'prev_attn_output')

    flop_counter.reset()
    with torch.no_grad():
        pipe("a cat", num_inference_steps=15,guidance_scale=4.5)
    
    cached_flops = flop_counter.total_flops
    flop_counter.print_summary()

    savings = (baseline_flops - cached_flops) / baseline_flops * 100
    print(f"Calculated Savings: {savings:.2f}%")
    
    if cached_flops < baseline_flops:
        print("SUCCESS: Caching significantly reduced FLOPs.")
    else:
        print("FAILURE: Caching did not reduce FLOPs.")

if __name__ == "__main__":
    test_flops_comparison()
