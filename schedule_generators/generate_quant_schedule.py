import json
import re
import torch
from diffusers import StableDiffusion3Pipeline

def generate_systematic_schedule():
    print("Fetching exact timesteps from SD3.5...")
    # Load just the scheduler to avoid massive RAM overhead
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", 
        text_encoder=None, text_encoder_2=None, text_encoder_3=None, 
        transformer=None, vae=None
    )
    pipe.scheduler.set_timesteps(15)
    timesteps = [int(t.item()) for t in pipe.scheduler.timesteps]
    print(f"Exact timesteps to be used as keys: {timesteps}")

    with open('layer_metrics.json', 'r') as f:
        metrics = json.load(f)
        
    schedule = {}
    
    # 1. Parse max Kappas per block from the metrics
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

    # 2. Build the Systematic Checkerboard Schedule
    for step_idx, t in enumerate(timesteps):
        t_str = str(t)
        schedule[t_str] = {}
        
        is_late_timestep = t < 300 # Fisher threshold
        
        for block_name, data in block_data.items():
            block_idx = data['idx']
            attn_k = data['attn_kappa']
            mlp_k = data['mlp_kappa']
            
            # --- CACHING LOGIC (Systematic 50%) ---
            if step_idx == 0:
                # First step MUST compute everything to populate the cache buffers
                use_cache = False
            else:
                # Checkerboard: alternate caching odd/even blocks per timestep
                if step_idx % 2 == 1:
                    use_cache = (block_idx % 2 != 0) # Cache odds
                else:
                    use_cache = (block_idx % 2 == 0) # Cache evens
                    
            # --- QUANTIZATION LOGIC (SVD + Fisher) ---
            if block_idx == 0:
                attn_bits, mlp_bits = 16, 16
                use_cache = False # Don't cache block 0
            else:
                # Attention Bits
                if attn_k > 500000: attn_bits = 16
                else: attn_bits = 8
                
                # MLP Bits
                if is_late_timestep: mlp_bits = 4
                elif mlp_k > 45: mlp_bits = 8
                else: mlp_bits = 4
                
            schedule[t_str][block_name] = {
                "cache": use_cache,
                "attn_bits": attn_bits,
                "mlp_bits": mlp_bits
            }

    with open('systematic_cache_schedule.json', 'w') as f:
        json.dump(schedule, f, indent=4)
        
    print(f"Generated systematic_cache_schedule.json with perfectly matched timesteps!")

if __name__ == "__main__":
    generate_systematic_schedule()