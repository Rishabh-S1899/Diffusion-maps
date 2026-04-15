# import json
# import numpy as np

# # Load your summary
# with open('analysis/summary_statistics.json', 'r') as f:
#     summary = json.load(f)

# # Collect all entropy values
# all_entropies = []
# for layer, timesteps in summary.items():
#     for timestep, metrics in timesteps.items():
#         all_entropies.append(metrics['entropy_mean'])

# print("Entropy statistics:")
# print(f"  Min:    {np.min(all_entropies):.3f}")
# print(f"  Max:    {np.max(all_entropies):.3f}")
# print(f"  Mean:   {np.mean(all_entropies):.3f}")
# print(f"  Median: {np.median(all_entropies):.3f}")
# print(f"  25th percentile: {np.percentile(all_entropies, 25):.3f}")
# print(f"  75th percentile: {np.percentile(all_entropies, 75):.3f}")


import json
from pathlib import Path

# Find first statistics.json file
stats_files = list(Path('outputs_parti-prompts').rglob('statistics.json'))
if stats_files:
    with open(stats_files[0], 'r') as f:
        data = json.load(f)
    
    print(f"Analyzing: {stats_files[0]}")
    print(f"\nTop-level keys (timesteps): {list(data.keys())}")
    print(f"Number of timesteps: {len(data.keys())}")
    
    # Show first timestep structure
    first_timestep = list(data.keys())[0]
    print(f"\nFirst timestep: {first_timestep}")
    print(f"Layers at this timestep: {list(data[first_timestep].keys())}")
    print(f"\nSample data:")
    first_layer = list(data[first_timestep].keys())[0]
    print(f"{first_layer}: {data[first_timestep][first_layer]}")