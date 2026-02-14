import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse

def load_and_aggregate(base_dir):
    """Aggregates all 447+ samples to find mean and std dev"""
    print(f"Scanning {base_dir} for statistics...")
    data_store = defaultdict(lambda: defaultdict(lambda: {'sim': [], 'ent': []}))
    
    found_files = list(Path(base_dir).rglob('statistics.json'))
    print(f"Found {len(found_files)} samples.")

    for f_path in found_files:
        try:
            with open(f_path, 'r') as f:
                sample_data = json.load(f)
                for ts, layers in sample_data.items():
                    ts_int = str(int(float(ts)))
                    for layer, metrics in layers.items():
                        # We use cross-attn metrics as the primary proxy for block stability
                        data_store[ts_int][layer]['sim'].append(metrics['similarity'])
                        data_store[ts_int][layer]['ent'].append(metrics['entropy'])
        except Exception as e:
            continue

    return data_store

def generate_optimal_schedule(data_store, output_path, sim_thresh=0.95, ent_thresh=2.0, safety_factor=1.5):
    schedule = {}
    
    print("Computing optimal strategies with safety margins...")
    for ts, layers in data_store.items():
        schedule[ts] = {}
        for layer, vals in layers.items():
            sims = np.array(vals['sim'])
            ents = np.array(vals['ent'])
            
            # Use 'Safe' metrics (Mean shifted by StdDev)
            # Higher safety_factor = more conservative (fewer skips, higher quality)
            safe_sim = np.mean(sims) - (safety_factor * np.std(sims))
            safe_ent = np.mean(ents) + (safety_factor * np.std(ents))
            
            do_cache = safe_sim > sim_thresh
            do_top_k = safe_ent < ent_thresh
            
            # Clean layer name (remove .attn suffix if present)
            clean_layer = layer.replace('.attn', '')
            
            schedule[ts][clean_layer] = {
                "cache": bool(do_cache),
                "top_k": bool(do_top_k)
            }
            
    with open(output_path, 'w') as f:
        json.dump(schedule, f, indent=2)
    print(f"Optimal schedule saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="outputs_parti-prompts", help="Directory containing prompt folders")
    parser.add_argument("--output", type=str, default="optimal_schedule.json")
    parser.add_argument("--sim", type=float, default=0.92, help="Lower-bound similarity threshold")
    parser.add_argument("--ent", type=float, default=2.5, help="Upper-bound entropy threshold")
    parser.add_argument("--safety", type=float, default=1.0, help="Safety multiplier for StdDev")
    args = parser.parse_args()
    
    data = load_and_aggregate(args.data_dir)
    if data:
        generate_optimal_schedule(data, args.output, args.sim, args.ent, args.safety)
    else:
        print("No data found to analyze.")
