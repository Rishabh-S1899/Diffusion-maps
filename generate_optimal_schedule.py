import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse

def load_and_aggregate(base_dir):
    print(f"Scanning {base_dir} for statistics...")
    data_store = defaultdict(lambda: defaultdict(lambda: {'sim': [], 'ent': []}))
    
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
                        data_store[ts_int][layer]['ent'].append(metrics['entropy'])
        except: continue
    return data_store

def generate_optimal_schedule(data_store, output_path, sim_thresh, ent_thresh, safety_factor):
    schedule = {}
    all_sims, all_ents = [], []
    
    # Pre-scan for distribution
    for ts in data_store:
        for layer in data_store[ts]:
            all_sims.extend(data_store[ts][layer]['sim'])
            all_ents.extend(data_store[ts][layer]['ent'])
    
    print("\n--- Data Distribution Analysis ---")
    print(f"Similarity: Min={np.min(all_sims):.3f}, Max={np.max(all_sims):.3f}, Mean={np.mean(all_sims):.3f}")
    print(f"Entropy:    Min={np.min(all_ents):.3f}, Max={np.max(all_ents):.3f}, Mean={np.mean(all_ents):.3f}")
    print(f"Current Thresholds: Sim > {sim_thresh}, Ent < {ent_thresh} (Safety: {safety_factor}x)")

    stats_summary = {"cache": 0, "top_k": 0, "both": 0, "total": 0}

    for ts, layers in data_store.items():
        schedule[ts] = {}
        for layer, vals in layers.items():
            sims, ents = np.array(vals['sim']), np.array(vals['ent'])
            
            # Rigorous Safety Calculation
            safe_sim = np.mean(sims) - (safety_factor * np.std(sims))
            safe_ent = np.mean(ents) + (safety_factor * np.std(ents))
            
            do_cache = safe_sim > sim_thresh
            do_top_k = safe_ent < ent_thresh
            
            clean_layer = layer.replace('.attn', '')
            schedule[ts][clean_layer] = {"cache": bool(do_cache), "top_k": bool(do_top_k)}
            
            stats_summary["total"] += 1
            if do_cache and do_top_k: stats_summary["both"] += 1
            elif do_cache: stats_summary["cache"] += 1
            elif do_top_k: stats_summary["top_k"] += 1
            
    with open(output_path, 'w') as f:
        json.dump(schedule, f, indent=2)
    
    print("\n--- Schedule Strategy Summary ---")
    print(f"Full Compute:   {stats_summary['total'] - (stats_summary['cache'] + stats_summary['top_k'] + stats_summary['both'])}")
    print(f"Pure Caching:   {stats_summary['cache']}")
    print(f"Pure Top-K:     {stats_summary['top_k']}")
    print(f"Rigorous Skip:  {stats_summary['both']} (Both active)")
    print(f"\nOptimal schedule saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="outputs_parti-prompts")
    parser.add_argument("--sim", type=float, default=0.95, help="Caching threshold (Higher = Safer)")
    parser.add_argument("--ent", type=float, default=1.2, help="Top-K threshold (Lower = Safer)")
    parser.add_argument("--safety", type=float, default=1.0)
    args = parser.parse_args()
    
    data = load_and_aggregate(args.data_dir)
    if data: generate_optimal_schedule(data, "optimal_schedule.json", args.sim, args.ent, args.safety)
