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
def load_all_statistics(base_dir='outputs_parti-prompts'):
    """Load statistics from all prompt folders"""
    all_stats = []
    
    base_path = Path(base_dir)
    
    # Debug: Check if directory exists
    if not base_path.exists():
        print(f"ERROR: Directory '{base_dir}' does not exist!")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Please check your directory structure.")
        return []
    
    # Look for statistics.json in subdirectories
    found_files = list(base_path.rglob('statistics.json'))
    print(f"Found {len(found_files)} statistics.json files")
    
    for stats_file in found_files:
        try:
            with open(stats_file, 'r') as f:
                data = json.load(f)
                
                # Get parent directory name as prompt identifier
                prompt_name = stats_file.parent.parent.name
                
                all_stats.append({
                    'prompt_name': prompt_name,
                    'stats': data
                })
                print(f"  ✓ Loaded: {prompt_name}")
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
    
    # Structure: {layer: {timestep: {'similarity': [values], 'entropy': [values]}}}
    aggregated = defaultdict(lambda: defaultdict(lambda: {'similarity': [], 'entropy': []}))
    
    for prompt_data in all_stats:
        stats = prompt_data['stats']
        for timestep_str, layers in stats.items():
            timestep = float(timestep_str)  # Convert "1000.0" to 1000.0
            for layer_name, metrics in layers.items():
                aggregated[layer_name][timestep]['similarity'].append(metrics['similarity'])
                aggregated[layer_name][timestep]['entropy'].append(metrics['entropy'])
    
    # Compute summary statistics
    summary = {}
    print(f"This is aggregated vals: {list(aggregated.items())[:10]}")  # Debug: show some layer keys
    for layer, timesteps in aggregated.items():
        summary[layer] = {}
        for timestep, metrics in timesteps.items():
            summary[layer][timestep] = {
                'similarity_mean': np.mean(metrics['similarity']),
                'similarity_std': np.std(metrics['similarity']),
                'entropy_mean': np.mean(metrics['entropy']),
                'entropy_std': np.std(metrics['entropy']),
                'num_samples': len(metrics['similarity'])
            }
    
    return summary, aggregated

# ==================== 3. VISUALIZATIONS ====================
def plot_entropy_heatmap(summary, output_dir='analysis'):
    """Heatmap showing entropy across layers and timesteps"""
    
    if not summary:
        print("⚠ Skipping entropy heatmap (no data)")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    layers = sorted(summary.keys())
    timesteps = sorted(list(summary[layers[0]].keys()), reverse=True)  # Reverse: 1000 → 0
    print(f"Thsi is layers {layers}")
    print(f"This is timesteps {timesteps}")
    entropy_matrix = np.zeros((len(layers), len(timesteps)))
    for i, layer in enumerate(layers):
        for j, timestep in enumerate(timesteps):
            entropy_matrix[i, j] = summary[layer][timestep]['entropy_mean']
    print(entropy_matrix)
    # Create better labels
    timestep_labels = [f"t{i}" for i in range(len(timesteps))]
    
    # Make figure much wider for readability
    fig_width = max(16, len(timesteps) * 1.2)  # Scale width based on number of timesteps
    plt.figure(figsize=(fig_width, 10))
    
    sns.heatmap(
        entropy_matrix,
        xticklabels=timestep_labels,
        yticklabels=[f"L{i}" for i in range(len(layers))],
        cmap='YlOrRd',
        annot=False,
        cbar_kws={'label': 'Mean Entropy'},
        fmt='.2f'
    )
    
    # Rotate x-axis labels and adjust spacing
    plt.xticks(rotation=0, ha='center')
    plt.xlabel('Denoising Step', fontsize=12)
    plt.ylabel('Layer', fontsize=12)
    plt.title('Attention Entropy\n(Red = Diffuse/High Entropy, Yellow = Focused/Low Entropy)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/entropy_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved entropy heatmap")


def plot_similarity_heatmap(summary, output_dir='analysis'):
    """Heatmap showing similarity across layers and timesteps"""
    
    if not summary:
        print("⚠ Skipping similarity heatmap (no data)")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sorted layers and timesteps
    layers = sorted(summary.keys())
    timesteps = sorted(list(summary[layers[0]].keys()), reverse=True)  # Reverse: 1000 → 0
    
    # Build matrix
    similarity_matrix = np.zeros((len(layers), len(timesteps)))
    for i, layer in enumerate(layers):
        for j, timestep in enumerate(timesteps):
            similarity_matrix[i, j] = summary[layer][timestep]['similarity_mean']
    
    # Create better labels (show step number instead of float value)
    timestep_labels = [f"t{i}" for i in range(len(timesteps))]  # t0, t1, t2...
    
    # Make figure much wider for readability
    fig_width = max(16, len(timesteps) * 1.2)
    plt.figure(figsize=(fig_width, 10))
    
    sns.heatmap(
        similarity_matrix,
        xticklabels=timestep_labels,  # Use step labels instead of float values
        yticklabels=[f"L{i}" for i in range(len(layers))],
        cmap='RdYlGn',
        vmin=0, vmax=1,
        annot=False,
        cbar_kws={'label': 'Mean Similarity'},
        fmt='.2f'
    )
    
    # Adjust x-axis labels
    plt.xticks(rotation=0, ha='center')
    plt.xlabel('Denoising Step (t0=pure noise → t14=clean image)', fontsize=12)
    plt.ylabel('Layer', fontsize=12)
    plt.title('Attention Map Similarity\n(Green = High Reuse Potential)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/similarity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved similarity heatmap")


def plot_temporal_evolution(summary, output_dir='analysis'):
    """Line plots showing how similarity/entropy evolve across timesteps"""
    
    if not summary:
        print("⚠ Skipping temporal evolution (no data)")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    layers = sorted(summary.keys())
    timesteps = sorted(list(summary[layers[0]].keys()), reverse=True)  # Reverse
    
    # Create step indices for x-axis (0, 1, 2... instead of 1000, 928, 857...)
    step_indices = list(range(len(timesteps)))
    
    # Collect all entropies for adaptive threshold
    all_entropies = []
    for layer in layers:
        for t in timesteps:
            if summary[layer][t]['similarity_mean'] > 0:
                all_entropies.append(summary[layer][t]['entropy_mean'])
    high_entropy_threshold = np.percentile(all_entropies, 75)
    
    # Select representative layers
    layer_indices = [0, len(layers)//3, 2*len(layers)//3, len(layers)-1]
    selected_layers = [layers[i] for i in layer_indices]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Similarity evolution
    for layer in selected_layers:
        sim_means = [summary[layer][t]['similarity_mean'] for t in timesteps]
        sim_stds = [summary[layer][t]['similarity_std'] for t in timesteps]
        
        layer_num = layers.index(layer)
        ax1.plot(step_indices, sim_means, marker='o', label=f'Layer {layer_num}', linewidth=2)
        ax1.fill_between(step_indices,
                         np.array(sim_means) - np.array(sim_stds),
                         np.array(sim_means) + np.array(sim_stds),
                         alpha=0.2)
    
    ax1.axhline(y=0.92, color='green', linestyle='--', alpha=0.5, label='High similarity threshold')
    ax1.axhline(y=0.75, color='orange', linestyle='--', alpha=0.5, label='Medium similarity threshold')
    ax1.set_xlabel('Denoising Step', fontsize=12)
    ax1.set_ylabel('Similarity', fontsize=12)
    ax1.set_title('Similarity Evolution Across Denoising (t0=pure noise → t14=clean)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Entropy evolution
    for layer in selected_layers:
        ent_means = [summary[layer][t]['entropy_mean'] for t in timesteps]
        ent_stds = [summary[layer][t]['entropy_std'] for t in timesteps]
        
        layer_num = layers.index(layer)
        ax2.plot(step_indices, ent_means, marker='o', label=f'Layer {layer_num}', linewidth=2)
        ax2.fill_between(step_indices,
                         np.array(ent_means) - np.array(ent_stds),
                         np.array(ent_means) + np.array(ent_stds),
                         alpha=0.2)
    
    ax2.axhline(y=high_entropy_threshold, color='blue', linestyle='--', alpha=0.5, 
                label=f'High entropy threshold ({high_entropy_threshold:.2f})')
    ax2.set_xlabel('Denoising Step', fontsize=12)
    ax2.set_ylabel('Entropy', fontsize=12)
    ax2.set_title('Entropy Evolution Across Denoising', fontsize=14, fontweight='bold')
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
    
    for layer, timesteps in summary.items():
        for timestep, metrics in timesteps.items():
            if metrics['similarity_mean'] == 0.0:
                continue
            similarities.append(metrics['similarity_mean'])
            entropies.append(metrics['entropy_mean'])
            timesteps_list.append(timestep)
    
    # Adaptive thresholds
    high_entropy_threshold = np.percentile(entropies, 75)
    medium_entropy_threshold = np.percentile(entropies, 50)
    
    plt.figure(figsize=(14, 10))
    scatter = plt.scatter(similarities, entropies,
                         c=timesteps_list, cmap='viridis',
                         alpha=0.5, s=30)
    plt.colorbar(scatter, label='Timestep')
    
    # Decision boundaries (all strategies)
    plt.axvline(x=0.92, color='green', linestyle='--', linewidth=2, label='High sim (0.92)')
    plt.axvline(x=0.75, color='orange', linestyle='--', linewidth=2, label='Med sim (0.75)')
    plt.axvline(x=0.60, color='brown', linestyle='--', linewidth=1.5, label='Low sim (0.60)')
    plt.axhline(y=high_entropy_threshold, color='blue', linestyle='--', linewidth=2, 
                label=f'High ent ({high_entropy_threshold:.2f})')
    plt.axhline(y=medium_entropy_threshold, color='cyan', linestyle='--', linewidth=1.5,
                label=f'Med ent ({medium_entropy_threshold:.2f})')
    
    # Get axis limits for annotation placement
    y_min, y_max = plt.ylim()
    x_min, x_max = plt.xlim()
    
    y_high = high_entropy_threshold + (y_max - high_entropy_threshold) * 0.4
    y_med = medium_entropy_threshold + (high_entropy_threshold - medium_entropy_threshold) * 0.5
    y_low = y_min + (medium_entropy_threshold - y_min) * 0.5
    
    # Annotate regions (all 4 strategies)
    # Top right: REUSE
    plt.text(0.96, y_high, 'REUSE\n(Str 2)', ha='center', fontsize=10, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Top middle-right: REUSE_SAFE
    plt.text(0.83, y_high, 'REUSE\nSAFE\n(Str 2)', ha='center', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Top left: SPARSIFY
    plt.text(0.68, y_high, 'SPARSIFY\n(Str 1)', ha='center', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # Middle region: QUANTIZE
    plt.text(0.78, y_med, 'QUANTIZE\n(Str 3)', ha='center', fontsize=9, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8))
    
    # Bottom region: FULL_COMPUTE
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
    summary, raw_data = aggregate_statistics(all_stats)
    
    # Save summary
    os.makedirs('analysis', exist_ok=True)
    with open('analysis/summary_statistics.json', 'w') as f:
        # Convert to serializable format
        serializable = {
            layer: {
                str(timestep): metrics
                for timestep, metrics in timesteps.items()
            }
            for layer, timesteps in summary.items()
        }
        json.dump(serializable, f, indent=2)
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