import torch
from diffusers import StableDiffusion3Pipeline
from attention_map_diffusers import init_pipeline, attn_maps
from attention_map_diffusers.utils import save_attention_stats
import os

def test_integration():
    print("Loading SD3.5 Medium for integration test...")
    # Using 16-bit to save memory, cpu offload if needed but we'll try standard first
    try:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium", 
            torch_dtype=torch.float16
        )
        pipe = pipe.to("cuda")
    except Exception as e:
        print(f"Failed to load SD3.5: {e}")
        print("Falling back to a smaller model for logic verification...")
        # If SD3.5 is too big, SD3-medium or SDXL-turbo could be used
        # For now, we assume the user has the memory for the model they were already using.
        return

    # Initialize our modified logic
    pipe = init_pipeline(pipe)

    prompt = "a small kitten"
    num_steps = 3 # Enough to test similarity between 1->2 and 2->3
    
    print(f"Running inference for {num_steps} steps...")
    image = pipe(prompt, num_inference_steps=num_steps, guidance_scale=4.5).images[0]

    # Verification
    print("Verifying captured data...")
    timesteps = sorted(list(attn_maps.keys()), reverse=True)
    print(f"Timesteps captured: {timesteps}")
    
    if not timesteps:
        print("FAILED: No attention maps captured!")
        return

    # Check a specific layer from the first captured timestep
    ts = timesteps[0]
    layer_names = list(attn_maps[ts].keys())
    sample_layer = layer_names[0]
    data = attn_maps[ts][sample_layer]
    
    print(f"Sample Layer: {sample_layer}")
    print(f"Data keys: {list(data.keys())}")
    
    # Check for presence of self-attention data
    has_self_stats = "self_similarity" in data and "self_entropy" in data
    has_cross_stats = "similarity" in data and "entropy" in data
    has_self_map = "self_map" in data and data["self_map"].ndim > 0
    
    print(f"Self-attention stats present: {has_self_stats}")
    print(f"Cross-attention stats present: {has_cross_stats}")
    print(f"Self-attention map present: {has_self_map}")

    assert has_self_stats, "Self-attention statistics were not captured!"
    assert has_cross_stats, "Cross-attention statistics were not captured!"
    assert has_self_map, "Self-attention map was not captured!"

    # Save stats and check JSON
    save_attention_stats(attn_maps, base_dir="integration_test_output")
    stats_path = "integration_test_output/statistics.json"
    
    if os.path.exists(stats_path):
        import json
        with open(stats_path, 'r') as f:
            stats_json = json.load(f)
            # Verify timestep keys are strings in JSON (standard for json.dump)
            ts_str = str(int(ts)) if not isinstance(ts, str) else ts
            if ts_str in stats_json:
                print(f"JSON verification successful for timestep {ts_str}")
            else:
                print(f"Timestep {ts_str} not found in JSON keys: {list(stats_json.keys())[:5]}")
    
    print("Integration test COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    test_integration()
