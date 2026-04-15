import json
import os
import numpy as np

def generate_fixed_schedule(summary_path='analysis/summary_statistics.json', output_file='fixed_schedule.json'):
    if not os.path.exists(summary_path):
        print(f"Error: Could not find {summary_path}")
        return

    with open(summary_path, 'r') as f:
        summary = json.load(f)

    # --- Threshold Configuration (Tuned based on your Heatmaps) ---
    THRESHOLDS = {
        'reuse_similarity': 0.90,       # Strict threshold for "Green Zone"
        'quantize_similarity': 0.65,    # Moderate threshold for "Yellow Zone"
        'high_entropy': 1.6,            # "Red Zone" in entropy map
        'low_entropy': 0.6              # "Yellow Zone" in entropy map
    }

    schedule = {}
    strategy_counts = {0: 0, 1: 0, 2: 0, 3: 0}

    # 1. Get all unique timesteps and layers
    layers = sorted(summary.keys())
    # Collect all timesteps
    all_timesteps = set()
    for l in layers:
        all_timesteps.update(summary[l].keys())
    
    # Sort descending (Diffusion process goes 1000 -> 0)
    timesteps = sorted(list(all_timesteps), key=lambda x: float(x), reverse=True)

    print(f"Generating schedule for {len(layers)} layers over {len(timesteps)} steps...")

    # 2. Build the Schedule Map
    for t_str in timesteps:
        # Convert to standard float for lookup if needed, but keep string key for JSON
        schedule[t_str] = {}
        
        for layer in layers:
            # Default to Full Compute (0) if data is missing
            if t_str not in summary[layer]:
                schedule[t_str][layer] = 0
                strategy_counts[0] += 1
                continue

            metrics = summary[layer][t_str]
            sim = metrics['similarity_mean']
            ent = metrics['entropy_mean']

            # --- DECISION LOGIC ---
            
            # RULE 1: First timestep (1000.0) must always be computed
            if float(t_str) >= 1000:
                strategy = 0 
            
            # RULE 2: High Similarity -> REUSE (Strategy 2)
            # (Matches the Green zones in your heatmap)
            elif sim >= THRESHOLDS['reuse_similarity']:
                strategy = 2
            
            # RULE 3: Low Similarity + High Entropy -> FULL COMPUTE / QUANTIZE
            # (Matches Layers 16-19 Red zones)
            elif ent > THRESHOLDS['high_entropy'] or sim >= THRESHOLDS['quantize_similarity']:
                # High entropy means diffuse attention (complex). 
                # We can Quantize (3) to save memory, or Full Compute (0) for safety.
                # Let's use Quantize as an optimization.
                strategy = 3 
            
            # RULE 4: Low Entropy -> SPARSIFY (Strategy 1)
            # (Matches Layer 1 Yellow zone)
            elif ent < THRESHOLDS['low_entropy']:
                strategy = 1
            # DEFAULT: Full Compute
            else:
                strategy = 0

            schedule[t_str][layer] = strategy
            strategy_counts[strategy] += 1

    # 3. Save to JSON
    with open(output_file, 'w') as f:
        json.dump(schedule, f, indent=2)

    print("\n" + "="*40)
    print(f"SCHEDULE GENERATED: {output_file}")
    print("="*40)
    total_ops = sum(strategy_counts.values())
    print(f"Strategy 2 (Reuse):    {strategy_counts[2]:4d} ({strategy_counts[2]/total_ops:.1%}) - [Aggressive Optimization]")
    print(f"Strategy 3 (Quantize): {strategy_counts[3]:4d} ({strategy_counts[3]/total_ops:.1%}) - [Memory Saving]")
    print(f"Strategy 1 (Sparsify): {strategy_counts[1]:4d} ({strategy_counts[1]/total_ops:.1%}) - [Compute Pruning]")
    print(f"Strategy 0 (Compute):  {strategy_counts[0]:4d} ({strategy_counts[0]/total_ops:.1%}) - [Baseline]")
    print("="*40)

if __name__ == "__main__":
    generate_fixed_schedule()