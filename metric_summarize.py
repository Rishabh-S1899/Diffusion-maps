import json

def summarize():
    try:
        with open('layer_metrics.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Could not find layer_metrics.json. Make sure it's in the same directory.")
        return

    # Flatten the JSON just in case the CLI nested it differently
    flat_data = {}
    def flatten(d, prefix=''):
        if isinstance(d, dict):
            # If we hit the actual metrics payload
            if any(k.lower() in ['kappa', 'condition_number'] for k in d.keys()):
                flat_data[prefix.strip('.')] = d
            else:
                for k, v in d.items():
                    flatten(v, prefix + k + '.')

    flatten(data)

    attn_kappas = []
    mlp_kappas = []
    fisher_stats = {}

    for layer_name, metrics in flat_data.items():
        name_lower = layer_name.lower()
        is_attn = 'attn' in name_lower or 'to_q' in name_lower or 'to_k' in name_lower or 'to_v' in name_lower or 'to_out' in name_lower
        is_mlp = 'ff' in name_lower or 'mlp' in name_lower
        
        # Grab Condition Number
        kappa = metrics.get('kappa', metrics.get('condition_number'))
        if kappa is not None:
            if is_attn:
                attn_kappas.append((layer_name, float(kappa)))
            elif is_mlp:
                mlp_kappas.append((layer_name, float(kappa)))

        # Grab Fisher Info
        for key, val in metrics.items():
            if 'fisher' in key.lower() and isinstance(val, (int, float)):
                if key not in fisher_stats:
                    fisher_stats[key] = []
                fisher_stats[key].append(float(val))

    print("\n" + "="*50)
    print(" 📊 LAYER METRICS SUMMARY FOR QUANTIZATION")
    print("="*50)

    # Calculate and print Attention stats
    if attn_kappas:
        max_attn = max(attn_kappas, key=lambda x: x[1])
        avg_attn = sum(x[1] for x in attn_kappas) / len(attn_kappas)
        print(f"\n[ATTENTION PROJECTIONS]")
        print(f"  Average Condition Number (κ): {avg_attn:.2f}")
        print(f"  Highest Condition Number (κ): {max_attn[1]:.2f} (Found in: {max_attn[0]})")
    else:
        print("\n[ATTENTION PROJECTIONS] No data found.")

    # Calculate and print MLP stats
    if mlp_kappas:
        max_mlp = max(mlp_kappas, key=lambda x: x[1])
        avg_mlp = sum(x[1] for x in mlp_kappas) / len(mlp_kappas)
        print(f"\n[MLP PROJECTIONS]")
        print(f"  Average Condition Number (κ): {avg_mlp:.2f}")
        print(f"  Highest Condition Number (κ): {max_mlp[1]:.2f} (Found in: {max_mlp[0]})")
    else:
        print("\n[MLP PROJECTIONS] No data found.")

    # Calculate and print Fisher stats
    print(f"\n[FISHER INFORMATION (SENSITIVITY)]")
    if fisher_stats:
        for t_key, values in fisher_stats.items():
            avg_fisher = sum(values) / len(values)
            max_fisher = max(values)
            print(f"  {t_key.upper()}:")
            print(f"    Average: {avg_fisher:.6f}  |  Max: {max_fisher:.6f}")
    else:
        print("  No Fisher Information found.")
    
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    summarize()