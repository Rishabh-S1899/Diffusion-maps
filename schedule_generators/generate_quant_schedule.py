import json
import re
import torch
from diffusers import StableDiffusion3Pipeline

def generate_protected_deepcache_schedule():
    print("Fetching exact 15 timesteps from SD3.5...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", 
        text_encoder=None, text_encoder_2=None, text_encoder_3=None, 
        transformer=None, vae=None
    )
    pipe.scheduler.set_timesteps(15)
    timesteps = [int(t.item()) for t in pipe.scheduler.timesteps]
    total_steps = len(timesteps)

    # Load SVD/Fisher metrics for quantization rules
    try:
        with open('layer_metrics.json', 'r') as f:
            metrics = json.load(f)
    except FileNotFoundError:
        print("layer_metrics.json not found. Quantization bits will default to 8/4.")
        metrics = {}
        
    schedule = {}
    
    # Extract Kappas for quantization
    block_data = {}
    for ts_key, layers in metrics.items():
        for layer_name, data in layers.items():
            match = re.search(r'transformer_blocks\.(\d+)', layer_name)
            if match:
                block_idx = int(match.group(1))
                block_name = f"transformer_blocks.{block_idx}"
                if block_name not in block_data:
                    block_data[block_name] = {'idx': block_idx, 'attn_kappa': 0, 'mlp_kappa': 0}
                kappa = data.get('kappa', data.get('condition_number', 0))
                if any(x in layer_name for x in ['attn', 'to_q', 'to_out']):
                    block_data[block_name]['attn_kappa'] = max(block_data[block_name]['attn_kappa'], kappa)
                elif any(x in layer_name for x in ['ff', 'mlp']):
                    block_data[block_name]['mlp_kappa'] = max(block_data[block_name]['mlp_kappa'], kappa)

    if not block_data:
        for i in range(38):
            block_data[f"transformer_blocks.{i}"] = {'idx': i, 'attn_kappa': 0, 'mlp_kappa': 0}

    # ARCHITECTURE DEFINITIONS
    fast_state_blocks = [0, 1, 2, 3, 4] + [33, 34, 35, 36, 37] # The "Bread" (Boundary blocks)
    
    for step_idx, t in enumerate(timesteps):
        t_str = str(t)
        schedule[t_str] = {}
        is_late_timestep = t < 300 
        
        # Define our temporal phases
        is_initial_phase = step_idx < 4
        is_final_phase = step_idx >= (total_steps - 2)
        
        # We start caching at step_idx 4. Compute on evens, cache on odds.
        update_slow_state = ((step_idx - 4) % 2 == 0)
        
        for block_name, data in block_data.items():
            block_idx = data['idx']
            attn_k = data['attn_kappa']
            mlp_k = data['mlp_kappa']
            
            # --- 1. CACHING LOGIC ---
            if is_initial_phase or is_final_phase:
                use_cache = False # Force compute everything in the protected zones
            elif block_idx in fast_state_blocks:
                use_cache = False # NEVER cache boundary blocks
            else:
                use_cache = not update_slow_state # Cache the middle blocks 50% of the time in the middle phase

            # --- 2. QUANTIZATION LOGIC ---
            # We still quantize the writes even if use_cache is False, to keep VRAM low!
            if block_idx in fast_state_blocks and (is_initial_phase or is_final_phase):
                 # Total high-precision safety for the absolute boundaries
                attn_bits, mlp_bits = 16, 16 
            else:
                if attn_k > 500000: attn_bits = 16
                else: attn_bits = 8
                
                if is_late_timestep: mlp_bits = 4
                elif mlp_k > 45: mlp_bits = 8
                else: mlp_bits = 4
                
            schedule[t_str][block_name] = {
                "cache": use_cache,
                "attn_bits": attn_bits,
                "mlp_bits": mlp_bits
            }

    with open('protected_cache_schedule.json', 'w') as f:
        json.dump(schedule, f, indent=4)
        
    print(f"Generated Protected DeepCache Schedule!")
    print("Steps 0-3: 100% Compute")
    print("Steps 4-12: DeepCache Sandwich (Alternating middle blocks)")
    print("Steps 13-14: 100% Compute")

if __name__ == "__main__":
    generate_protected_deepcache_schedule()