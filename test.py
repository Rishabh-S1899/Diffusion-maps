import json
import numpy as np

# Load your summary
with open('analysis/summary_statistics.json', 'r') as f:
    summary = json.load(f)

# Collect all entropy values
all_entropies = []
for layer, timesteps in summary.items():
    for timestep, metrics in timesteps.items():
        all_entropies.append(metrics['entropy_mean'])

print("Entropy statistics:")
print(f"  Min:    {np.min(all_entropies):.3f}")
print(f"  Max:    {np.max(all_entropies):.3f}")
print(f"  Mean:   {np.mean(all_entropies):.3f}")
print(f"  Median: {np.median(all_entropies):.3f}")
print(f"  25th percentile: {np.percentile(all_entropies, 25):.3f}")
print(f"  75th percentile: {np.percentile(all_entropies, 75):.3f}")