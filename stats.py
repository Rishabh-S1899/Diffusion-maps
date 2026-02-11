import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from pathlib import Path

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ==================== 1. LOAD DATA ====================
def load_all_statistics(base_dir='outputs/parti-prompts'):
    """Load statistics from all prompt folders"""
    all_stats = []
    
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"ERROR: Directory '{base_dir}' does not exist!")
        return []
    
    found_files = list(base_path.rglob('statistics.json'))
    print(f"Found {len(found_files)} statistics.json files")
    
    for stats_file in found_files:
        try:
            with open(stats_file, 'r') as f:
                data = json.load(f)
                
                prompt_name = stats_file.parent.parent.name
                
                # DEBUG: Check timesteps in raw data
                num_timesteps = len(data.keys())
                if num_timesteps != 15:
                    print(f"  ⚠ {prompt_name}: only {num_timesteps} timesteps!")
                
                all_stats.append({
                    'prompt_name': prompt_name,
                    'stats': data  # Keep original structure with string keys
                })
        except Exception as e:
            print(f"  ✗ Error loading {stats_file}: {e}")
    
    print(f"\nTotal loaded: {len(all_stats)} prompts\n")
    return all_stats

# ==================== 2. AGGREGATE STATISTICS ====================
def aggregate_statistics(all_stats):
    """Compute mean and std across all prompts"""
    
    if not all_stats:
        print("ERROR: No statistics to aggregate!")
        return {}, {}
    
    # Structure: {layer: {timestep_str: {'similarity': [values], 'entropy': [values]}}}
    aggregated = defaultdict(lambda: defaultdict(lambda: {'similarity': [], 'entropy': []}))
    
    for prompt_data in all_stats:
        stats = prompt_data['stats']
        for timestep_str, layers in stats.items():
            # KEEP AS STRING - don't convert to float yet
            for layer_name, metrics in layers.items():
                aggregated[layer_name][timestep_str]['similarity'].append(metrics['similarity'])
                aggregated[layer_name][timestep_str]['entropy'].append(metrics['entropy'])
    
    # Compute summary statistics
    summary = {}
    for layer, timesteps in aggregated.items():
        summary[layer] = {}
        for timestep_str, metrics in timesteps.items():
            summary[layer][timestep_str] = {
                'similarity_mean': np.mean(metrics['similarity']),
                'similarity_std': np.std(metrics['similarity']),
                'entropy_mean': np.mean(metrics['entropy']),
                'entropy_std': np.std(metrics['entropy']),
                'num_samples': len(metrics['similarity'])
            }
    
    return summary, aggregated

# ==================== 3. VISUALIZATIONS ====================
# ==================== 3. VISUALIZATIONS (FIXED) ====================
def plot_similarity_heatmap(summary, output_dir='analysis'):
    """Heatmap showing similarity across layers and timesteps"""
    
    if not summary:
        print("⚠ Skipping similarity heatmap (no data)")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Get all unique timesteps from ALL layers (Robust fix)
    all_timesteps = set()
    for layer in summary:
        all_timesteps.update(summary[layer].keys())
    
    # Sort timesteps descending (1000 -> 0)
    timesteps = sorted(list(all_timesteps), key=lambda x: float(x), reverse=True)
    
    # Sort layers naturally
    layers = sorted(summary.keys())
    
    print(f"Debug - Total unique timesteps found: {len(timesteps)}")
    
    # 2. Build matrix with NaN for missing values
    similarity_matrix = np.full((len(layers), len(timesteps)), np.nan)
    
    for i, layer in enumerate(layers):
        for j, t_str in enumerate(timesteps):
            if t_str in summary[layer]:
                similarity_matrix[i, j] = summary[layer][t_str]['similarity_mean']
    
    # Create labels
    # formatting float labels to avoid long strings like '975.07861328125'
    timestep_labels = [f"{float(t):.1f}" for t in timesteps]
    
    # Dynamic figure width
    fig_width = max(16, len(timesteps) * 1.2)
    plt.figure(figsize=(fig_width, 10))
    
    sns.heatmap(
        similarity_matrix,
        xticklabels=timestep_labels,
        yticklabels=layers, # Use actual layer names
        cmap='RdYlGn',
        vmin=0, vmax=1,
        annot=False,
        cbar_kws={'label': 'Mean Similarity'}
    )
    
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Denoising Step (High=Noise → Low=Clean)', fontsize=12)
    plt.ylabel('Layer', fontsize=12)
    plt.title('Attention Map Similarity\n(Green = High Reuse Potential)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/similarity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved similarity heatmap")


def plot_entropy_heatmap(summary, output_dir='analysis'):
    """Heatmap showing entropy across layers and timesteps"""
    
    if not summary:
        print("⚠ Skipping entropy heatmap (no data)")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Robust Timestep Collection
    all_timesteps = set()
    for layer in summary:
        all_timesteps.update(summary[layer].keys())
    timesteps = sorted(list(all_timesteps), key=lambda x: float(x), reverse=True)
    layers = sorted(summary.keys())
    
    # 2. Build Matrix
    entropy_matrix = np.full((len(layers), len(timesteps)), np.nan)
    for i, layer in enumerate(layers):
        for j, t_str in enumerate(timesteps):
            if t_str in summary[layer]:
                entropy_matrix[i, j] = summary[layer][t_str]['entropy_mean']
    
    timestep_labels = [f"{float(t):.1f}" for t in timesteps]
    
    fig_width = max(16, len(timesteps) * 1.2)
    plt.figure(figsize=(fig_width, 10))
    
    sns.heatmap(
        entropy_matrix,
        xticklabels=timestep_labels,
        yticklabels=layers,
        cmap='YlOrRd',
        annot=False,
        cbar_kws={'label': 'Mean Entropy'}
    )
    
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Denoising Step', fontsize=12)
    plt.ylabel('Layer', fontsize=12)
    plt.title('Attention Entropy\n(Red = Diffuse/High Entropy, Yellow = Focused/Low Entropy)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/entropy_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved entropy heatmap")


def plot_temporal_evolution(summary, output_dir='analysis'):
    """Line plots showing how similarity/entropy evolve across timesteps"""
    
    if not summary:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    layers = sorted(summary.keys())
    
    # Robust Timestep Collection
    all_timesteps = set()
    for layer in summary:
        all_timesteps.update(summary[layer].keys())
    timesteps = sorted(list(all_timesteps), key=lambda x: float(x), reverse=True)
    
    step_indices = list(range(len(timesteps)))
    
    # Collect entropies for threshold
    all_entropies = []
    for layer in layers:
        for t_str in timesteps:
            if t_str in summary[layer] and summary[layer][t_str]['similarity_mean'] > 0:
                all_entropies.append(summary[layer][t_str]['entropy_mean'])
    
    high_entropy_threshold = np.percentile(all_entropies, 75) if all_entropies else 0
    
    # Select representative layers (First, Middle, Last)
    if len(layers) > 4:
        layer_indices = [0, len(layers)//3, 2*len(layers)//3, len(layers)-1]
        selected_layers = [layers[i] for i in layer_indices]
    else:
        selected_layers = layers
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Similarity evolution
    for layer in selected_layers:
        sim_means = []
        sim_stds = []
        valid_indices = []
        
        for idx, t_str in enumerate(timesteps):
            if t_str in summary[layer]:
                sim_means.append(summary[layer][t_str]['similarity_mean'])
                sim_stds.append(summary[layer][t_str]['similarity_std'])
                valid_indices.append(idx)
        
        if valid_indices:
            layer_short = layer.split('.')[-2] if '.' in layer else layer
            ax1.plot(valid_indices, sim_means, marker='o', label=f'L{layer_short}', linewidth=2)
            ax1.fill_between(valid_indices,
                             np.array(sim_means) - np.array(sim_stds),
                             np.array(sim_means) + np.array(sim_stds),
                             alpha=0.2)
    
    ax1.set_xticks(step_indices)
    ax1.set_xticklabels([f"{float(t):.0f}" for t in timesteps], rotation=45)
    ax1.axhline(y=0.92, color='green', linestyle='--', alpha=0.5, label='Reuse Threshold')
    ax1.set_ylabel('Similarity')
    ax1.set_title('Similarity Evolution', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Entropy evolution
    for layer in selected_layers:
        ent_means = []
        ent_stds = []
        valid_indices = []
        
        for idx, t_str in enumerate(timesteps):
            if t_str in summary[layer]:
                ent_means.append(summary[layer][t_str]['entropy_mean'])
                ent_stds.append(summary[layer][t_str]['entropy_std'])
                valid_indices.append(idx)

        if valid_indices:
            layer_short = layer.split('.')[-2] if '.' in layer else layer
            ax2.plot(valid_indices, ent_means, marker='o', label=f'L{layer_short}', linewidth=2)
            ax2.fill_between(valid_indices,
                             np.array(ent_means) - np.array(ent_stds),
                             np.array(ent_means) + np.array(ent_stds),
                             alpha=0.2)
    
    ax2.set_xticks(step_indices)
    ax2.set_xticklabels([f"{float(t):.0f}" for t in timesteps], rotation=45)
    ax2.axhline(y=high_entropy_threshold, color='blue', linestyle='--', alpha=0.5, label='High Entropy')
    ax2.set_ylabel('Entropy')
    ax2.set_title('Entropy Evolution', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/temporal_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved temporal evolution plot")
def plot_optimization_scatter(summary, output_dir='analysis'):
    """Scatter plot showing optimization strategy space"""
    
    if not summary:
        print("⚠ Skipping optimization scatter (no data)")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    similarities = []
    entropies = []
    timesteps_list = []
    
    for layer, timesteps_dict in summary.items():
        for timestep_str, metrics in timesteps_dict.items():
            if metrics['similarity_mean'] == 0.0:
                continue
            similarities.append(metrics['similarity_mean'])
            entropies.append(metrics['entropy_mean'])
            timesteps_list.append(float(timestep_str))  # CONVERT TO FLOAT HERE
    
    # Adaptive thresholds
    high_entropy_threshold = np.percentile(entropies, 75)
    medium_entropy_threshold = np.percentile(entropies, 50)
    
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(similarities, entropies,
                         c=timesteps_list, cmap='viridis',  # Now these are floats
                         alpha=0.5, s=30)
    plt.colorbar(scatter, label='Timestep')
    
    # Decision boundaries
    plt.axvline(x=0.92, color='green', linestyle='--', linewidth=2, label='High sim (0.92)')
    plt.axvline(x=0.75, color='orange', linestyle='--', linewidth=2, label='Med sim (0.75)')
    plt.axvline(x=0.60, color='brown', linestyle='--', linewidth=1.5, label='Low sim (0.60)')
    plt.axhline(y=high_entropy_threshold, color='blue', linestyle='--', linewidth=2, 
                label=f'High ent ({high_entropy_threshold:.2f})')
    plt.axhline(y=medium_entropy_threshold, color='cyan', linestyle='--', linewidth=1.5,
                label=f'Med ent ({medium_entropy_threshold:.2f})')
    
    # Get axis limits
    y_min, y_max = plt.ylim()
    
    y_high = high_entropy_threshold + (y_max - high_entropy_threshold) * 0.4
    y_med = medium_entropy_threshold + (high_entropy_threshold - medium_entropy_threshold) * 0.5
    y_low = y_min + (medium_entropy_threshold - y_min) * 0.5
    
    # Annotate regions
    plt.text(0.96, y_high, 'REUSE\n(Str 2)', ha='center', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    plt.text(0.83, y_high, 'REUSE\nSAFE\n(Str 2)', ha='center', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    plt.text(0.68, y_high, 'SPARSIFY\n(Str 1)', ha='center', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    plt.text(0.78, y_med, 'QUANTIZE\n(Str 3)', ha='center', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8))
    plt.text(0.70, y_low, 'FULL\nCOMPUTE\n(Str 4)', ha='center', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    plt.xlabel('Similarity (mean across prompts)', fontsize=12)
    plt.ylabel('Entropy (mean across prompts)', fontsize=12)
    plt.title('Optimization Strategy Space - All 4 Strategies\n(Each point = one layer at one timestep)', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/optimization_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved optimization scatter plot")
# ==================== 4. QUANTITATIVE ANALYSIS ====================
def count_optimization_opportunities(summary):
    """Calculate percentage of operations that can be optimized"""
    
    if not summary:
        print("⚠ No data to analyze")
        return {}
    
    # First, collect all entropy values to set adaptive threshold
    all_entropies = []
    for layer, timesteps in summary.items():
        for timestep, metrics in timesteps.items():
            if metrics['similarity_mean'] > 0.0:
                all_entropies.append(metrics['entropy_mean'])
    
    # Use 75th percentile as "high entropy" threshold
    high_entropy_threshold = np.percentile(all_entropies, 75)
    medium_entropy_threshold = np.percentile(all_entropies, 50)
    
    print(f"\nAdaptive thresholds:")
    print(f"  Entropy range: [{np.min(all_entropies):.3f}, {np.max(all_entropies):.3f}]")
    print(f"  High entropy threshold (75th percentile): {high_entropy_threshold:.3f}")
    print(f"  Medium entropy threshold (median): {medium_entropy_threshold:.3f}")
    
    total = 0
    reuse = 0
    reuse_safe = 0
    sparsify = 0
    quantize = 0  # NEW: Strategy 3
    full_compute = 0
    
    for layer, timesteps in summary.items():
        for timestep, metrics in timesteps.items():
            # Skip first timestep
            if metrics['similarity_mean'] == 0.0:
                continue
            
            total += 1
            sim = metrics['similarity_mean']
            ent = metrics['entropy_mean']
            
            # Decision logic with all 4 strategies
            if sim > 0.92:
                # Strategy 2: REUSE - Very high similarity
                reuse += 1
                
            elif sim > 0.75 and ent > high_entropy_threshold:
                # Strategy 2: REUSE_SAFE - Moderate similarity + high entropy
                reuse_safe += 1
                
            elif ent > high_entropy_threshold:
                # Strategy 1: SPARSIFY - High entropy (diffuse attention)
                sparsify += 1
                
            elif sim > 0.60 or ent > medium_entropy_threshold:
                # Strategy 3: QUANTIZE - Moderate similarity or moderate entropy
                # Not safe to reuse/sparsify, but can reduce precision
                quantize += 1
                
            else:
                # Strategy 4: FULL_COMPUTE - Low similarity + low entropy
                # Must compute with full precision
                full_compute += 1
    
    print("\n" + "="*70)
    print("OPTIMIZATION OPPORTUNITIES (All 4 Strategies)")
    print("="*70)
    print(f"Total [layer, timestep] pairs analyzed: {total}")
    print(f"\nStrategy breakdown:")
    print(f"  Strategy 2 - REUSE (sim > 0.92):                    {reuse:4d} ({100*reuse/total:5.1f}%)")
    print(f"  Strategy 2 - REUSE_SAFE (sim>0.75, ent>{high_entropy_threshold:.2f}):  {reuse_safe:4d} ({100*reuse_safe/total:5.1f}%)")
    print(f"  Strategy 1 - SPARSIFY (ent > {high_entropy_threshold:.2f}):             {sparsify:4d} ({100*sparsify/total:5.1f}%)")
    print(f"  Strategy 3 - QUANTIZE (sim>0.6 or ent>{medium_entropy_threshold:.2f}):   {quantize:4d} ({100*quantize/total:5.1f}%)")
    print(f"  Strategy 4 - FULL_COMPUTE (else):                   {full_compute:4d} ({100*full_compute/total:5.1f}%)")
    print(f"\n{'='*70}")
    print(f"Total optimizable: {100*(total - full_compute)/total:.1f}%")
    print(f"  - Skip computation (reuse): {100*(reuse + reuse_safe)/total:.1f}%")
    print(f"  - Reduce computation (sparsify): {100*sparsify/total:.1f}%")
    print(f"  - Reduce precision (quantize): {100*quantize/total:.1f}%")
    print(f"{'='*70}\n")
    
    return {
        'reuse': reuse,
        'reuse_safe': reuse_safe,
        'sparsify': sparsify,
        'quantize': quantize,
        'full_compute': full_compute,
        'total': total,
        'high_entropy_threshold': high_entropy_threshold,
        'medium_entropy_threshold': medium_entropy_threshold
    }
def analyze_variance(summary):
    """Check if statistics are stable across prompts"""
    
    if not summary:
        return
    
    high_variance = []
    
    for layer, timesteps in summary.items():
        for timestep, metrics in timesteps.items():
            if metrics['similarity_mean'] == 0.0:
                continue
            
            # High std relative to mean indicates instability
            if metrics['similarity_std'] > 0.15:
                high_variance.append({
                    'layer': layer,
                    'timestep': int(timestep),
                    'sim_mean': metrics['similarity_mean'],
                    'sim_std': metrics['similarity_std']
                })
    
    print("\n" + "="*60)
    print("VARIANCE ANALYSIS")
    print("="*60)
    
    if high_variance:
        print(f"\nHigh variance cases: {len(high_variance)}")
        print("(These may need prompt-conditional strategies)\n")
        for case in sorted(high_variance, key=lambda x: x['sim_std'], reverse=True)[:10]:
            layer_num = case['layer'].split('.')[-2]
            print(f"  Layer {layer_num:3s} @ t={case['timestep']:4d}: "
                  f"μ={case['sim_mean']:.3f} ± σ={case['sim_std']:.3f}")
    else:
        print("\n✓ Low variance across all layers/timesteps")
        print("  Fixed schedule should generalize well!")
    
    print("="*60 + "\n")

# ==================== 5. MAIN EXECUTION ====================

if __name__ == '__main__':
    print("Starting analysis...\n")
    
    # Detect directory structure
    print("Checking directory structure...")
    if os.path.exists('outputs_parti-prompts'):
        base_dir = 'outputs_parti-prompts'
    elif os.path.exists('outputs'):
        base_dir = 'outputs'
    else:
        print("ERROR: Could not find 'outputs' directory!")
        print(f"Current directory: {os.getcwd()}")
        print("\nPlease run this script from the directory containing 'outputs/'")
        exit(1)
    
    print(f"Using base directory: {base_dir}\n")
    
    # Load data
    all_stats = load_all_statistics(base_dir)
    
    if not all_stats:
        print("\n❌ No data loaded. Please check:")
        print("   1. Are statistics.json files in the correct location?")
        print("   2. Directory structure should be:")
        print("      outputs/")
        print("        └── prompt_folder_1/")
        print("            └── attn_stats/")
        print("                └── statistics.json")
        exit(1)
    
    # Aggregate
    print("Aggregating statistics...")
    # In main execution, after aggregation:
    summary, raw_data = aggregate_statistics(all_stats)

    # DEBUG: Check what's in summary
    print("\n=== DEBUG SUMMARY ===")
    first_layer = list(summary.keys())[0]
    print(f"First layer: {first_layer}")
    print(f"Timesteps in summary: {list(summary[first_layer].keys())}")
    print(f"Number of timesteps: {len(summary[first_layer].keys())}")
    print("=" * 50 + "\n")
    
    # Save summary
    # Save summary (summary already has string keys)
    os.makedirs('analysis', exist_ok=True)
    with open('analysis/summary_statistics.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("✓ Saved summary statistics\n")
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_similarity_heatmap(summary)
    plot_entropy_heatmap(summary)
    plot_temporal_evolution(summary)
    plot_optimization_scatter(summary)
    
    # Quantitative analysis
    opportunities = count_optimization_opportunities(summary)
    analyze_variance(summary)
    
    print("\n✓ Analysis complete!")
    print(f"  - Plots saved to: analysis/*.png")
    print(f"  - Summary saved to: analysis/summary_statistics.json")