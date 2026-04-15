import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse

def load_and_aggregate(base_dir):
    print(f"Scanning {base_dir} for statistics...")
    data_store = defaultdict(lambda: defaultdict(lambda: {'sim': []}))
    
    found_files = list(Path(base_dir).rglob('statistics.json'))
    if not found_files:
        return None

    for f_path in found_files:
        try:
            with open(f_path, 'r') as f:
                sample_data = json.load(f)
                for ts, layers in sample_data.items():
                    ts_int = str(int(float(ts)))
                    for layer, metrics in layers.items():
                        data_store[ts_int][layer]['sim'].append(metrics['similarity'])
        except: continue
    return data_store

def generate_optimal_schedule(data_store, output_path, sim_thresh, safety_factor):
    schedule = {}
    all_sims = []
    
    for ts in data_store:
        for layer in data_store[ts]:
            all_sims.extend(data_store[ts][layer]['sim'])
    
    print("\n--- Data Distribution Analysis ---")
    print(f"Similarity: Min={np.min(all_sims):.3f}, Max={np.max(all_sims):.3f}, Mean={np.mean(all_sims):.3f}")
    print(f"Current Threshold: Sim > {sim_thresh} (Safety: {safety_factor}x)")

    stats_summary = {"cache": 0, "total": 0}

    for ts, layers in data_store.items():
        schedule[ts] = {}
        for layer, vals in layers.items():
            sims = np.array(vals['sim'])
            
            # Rigorous Safety Calculation: Mean - (Safety * StdDev)
            safe_sim = np.mean(sims) - (safety_factor * np.std(sims))
            do_cache = safe_sim > sim_thresh
            
            clean_layer = layer.replace('.attn', '')
            schedule[ts][clean_layer] = bool(do_cache)
            
            stats_summary["total"] += 1
            if do_cache: stats_summary["cache"] += 1
            
    with open(output_path, 'w') as f:
        json.dump(schedule, f, indent=2)
    
    print("\n--- Schedule Strategy Summary ---")
    print(f"Full Compute:   {stats_summary['total'] - stats_summary['cache']}")
    print(f"Cached Steps:   {stats_summary['cache']} ({100*stats_summary['cache']/stats_summary['total']:.1f}%)")
    print(f"\nOptimal schedule saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="outputs_parti-prompts")
    parser.add_argument("--sim", type=float, default=0.95, help="Caching threshold (Higher = Safer)")
    parser.add_argument("--safety", type=float, default=1.0, help="StdDev multiplier for safety margin")
    parser.add_argument("--output", type=str, default="optimal_schedule.json")
    args = parser.parse_args()
    
    data = load_and_aggregate(args.data_dir)
    if data: generate_optimal_schedule(data, args.output, args.sim, args.safety)
