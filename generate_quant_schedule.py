import json

def generate_adaptive_schedule():
    with open('layer_metrics.json', 'r') as f:
        metrics = json.load(f)
        
    schedule = {}
    
    # We will simulate a standard 15-step inference schedule (1000 to 0)
    # You can adjust this list to match your actual timesteps
    timesteps = [1000, 928, 857, 785, 714, 642, 571, 500, 428, 357, 285, 214, 142, 71, 35]
    
    # Flatten metrics to easily find block data
    block_data = {}
    for ts_key, layers in metrics.items():
        for layer_name, data in layers.items():
            # Extract block index (e.g., "transformer_blocks.4" -> "4")
            parts = layer_name.split('.')
            block_idx = None
            for i, p in enumerate(parts):
                if p == 'transformer_blocks' and i+1 < len(parts):
                    block_idx = parts[i+1]
                    break
            
            if block_idx is not None:
                block_name = f"transformer_blocks.{block_idx}"
                if block_name not in block_data:
                    block_data[block_name] = {'attn_kappa': 0, 'mlp_kappa': 0}
                
                kappa = data.get('kappa', 0)
                if 'attn' in layer_name or 'to_q' in layer_name or 'to_out' in layer_name:
                    block_data[block_name]['attn_kappa'] = max(block_data[block_name]['attn_kappa'], kappa)
                elif 'ff' in layer_name or 'mlp' in layer_name:
                    block_data[block_name]['mlp_kappa'] = max(block_data[block_name]['mlp_kappa'], kappa)

    # Build the schedule per timestep
    for t in timesteps:
        t_str = str(t)
        schedule[t_str] = {}
        
        is_late_timestep = t < 300
        
        for block_name, kappas in block_data.items():
            # RULE 1: Block 0 is highly sensitive (Fisher spike)
            if block_name == "transformer_blocks.0":
                schedule[t_str][block_name] = {"cache": False, "attn_bits": 16, "mlp_bits": 16}
                continue
                
            attn_k = kappas['attn_kappa']
            mlp_k = kappas['mlp_kappa']
            
            # RULE 2: Attention Logic
            if attn_k > 500000:
                attn_bits = 16 # Extreme spike, don't quantize
            else:
                attn_bits = 8  # Safe zone for 8-bit channel-wise
                
            # RULE 3 & 4: MLP Logic
            if is_late_timestep:
                mlp_bits = 4   # Force 4-bit late in generation
            elif mlp_k > 45:
                mlp_bits = 8   # Mild spike, use 8-bit
            else:
                mlp_bits = 4   # Safe zone, aggressively compress
                
            # If both are 16-bit, caching might not be worth the VRAM read/write
            cache_enabled = not (attn_bits == 16 and mlp_bits == 16)
                
            schedule[t_str][block_name] = {
                "cache": cache_enabled,
                "attn_bits": attn_bits,
                "mlp_bits": mlp_bits
            }

    with open('cache_schedule.json', 'w') as f:
        json.dump(schedule, f, indent=4)
        
    print("Successfully generated adaptive cache_schedule.json!")
    print(f"Total blocks scheduled: {len(block_data)} across {len(timesteps)} timesteps.")

if __name__ == "__main__":
    generate_adaptive_schedule()