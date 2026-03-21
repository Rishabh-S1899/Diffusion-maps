import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random
from diffusers import StableDiffusion3Pipeline
from datasets import load_dataset
from tqdm import tqdm

def categorize_layer(name):
    """Categorize layers into Attention or MLP based on their name."""
    name_lower = name.lower()
    if any(x in name_lower for x in ['to_q', 'to_k', 'to_v', 'to_out', 'add_q_proj', 'add_k_proj', 'add_v_proj', 'to_add_out', 'attn']):
        return "Attention"
    if any(x in name_lower for x in ['ff', 'feed_forward', 'proj_out', 'proj_in']):
        return "MLP"
    return "Other"

def get_linear_layers(model):
    """Find all nn.Linear layers and categorize them."""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layers.append({'name': name, 'module': module, 'category': categorize_layer(name)})
    return layers

def profile_weights(layers):
    """Calculate SVD-based metrics for each layer."""
    print("Profiling weights (SVD)...")
    metrics = {}
    for layer_info in tqdm(layers):
        name = layer_info['name']
        module = layer_info['module']
        weight = module.weight.data.detach().float()
        try:
            singular_values = torch.linalg.svdvals(weight)
            s_max, s_min = singular_values[0].item(), singular_values[-1].item()
            metrics[name] = {
                'category': layer_info['category'],
                'singular_values': singular_values.cpu().numpy().tolist(),
                'spectral_norm': s_max,
                'condition_number': s_max / (s_min + 1e-10)
            }
        except Exception as e:
            print(f"Error profiling layer {name}: {e}")
    return metrics

def calculate_fisher(pipe, layers, device, prompts, timesteps):
    """Calculate Fisher Info across multiple prompts and timesteps."""
    print(f"Calculating Fisher Info for {len(prompts)} prompts and {len(timesteps)} timesteps...")
    model = pipe.transformer
    model.train()
    for param in model.parameters(): param.requires_grad = True
    
    batch_size = 1
    # Initialize storage: {layer_name: {timestep: [values_across_prompts]}}
    raw_fisher = {l['name']: {t: [] for t in timesteps} for l in layers}
    
    for i, prompt in enumerate(prompts):
        print(f"  Prompt {i+1}/{len(prompts)}: {prompt[:40]}...")
        try:
            with torch.no_grad():
                prompt_embeds, _, pooled_prompt_embeds, _ = pipe.encode_prompt(
                    prompt=prompt, prompt_2=prompt, prompt_3=prompt
                )
                encoder_hidden_states = prompt_embeds.to(device=device, dtype=torch.bfloat16)
                pooled_projections = pooled_prompt_embeds.to(device=device, dtype=torch.bfloat16)
                hidden_states = torch.randn(batch_size, 16, 64, 64, device=device, dtype=torch.bfloat16)
        except Exception as e:
            print(f"    Error encoding prompt {i}: {e}")
            continue

        for t_val in tqdm(timesteps, leave=False, desc="Timesteps"):
            timestep = torch.tensor([t_val], device=device, dtype=torch.bfloat16)
            model.zero_grad(set_to_none=True)
            
            output = model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                return_dict=True
            ).sample
            
            loss = output.pow(2).mean()
            loss.backward()
            
            for layer_info in layers:
                name = layer_info['name']
                if layer_info['module'].weight.grad is not None:
                    fisher = layer_info['module'].weight.grad.pow(2).mean().item()
                    raw_fisher[name][t_val].append(fisher)
            
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

        # Periodic Save Checkpoint
        if (i + 1) % 10 == 0:
            print(f"    Intermediate checkpoint saved at prompt {i+1}...")
            with open("raw_fisher_checkpoint.json", 'w') as f:
                # We save raw values so we can re-calculate stats if needed
                json.dump(raw_fisher, f)

    # Aggregate statistics into a structured format
    fisher_results = {}
    for name in raw_fisher:
        fisher_results[name] = {}
        for t in timesteps:
            vals = np.array(raw_fisher[name][t])
            if len(vals) > 0:
                fisher_results[name][t] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals))
                }
            
    return fisher_results

def export_json(weight_metrics, fisher_results, output_path, num_samples, timesteps):
    """Merge everything into a master scientific JSON."""
    combined = {
        "Attention": {}, 
        "MLP": {}, 
        "Other": {}, 
        "metadata": {
            "num_samples": num_samples,
            "timesteps": timesteps,
            "model_architecture": "SD3.5-MM-DiT"
        }
    }
    
    for name in weight_metrics:
        category = weight_metrics[name].get('category', 'Other')
        # Combine static weight metrics with the dynamic Fisher stats
        entry = {
            "name": name,
            "category": category,
            "spectral_norm": weight_metrics[name]["spectral_norm"],
            "condition_number": weight_metrics[name]["condition_number"],
            "singular_values": weight_metrics[name]["singular_values"],
            "fisher_stats": fisher_results.get(name, {})
        }
        combined[category][name] = entry
        
    with open(output_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"Master scientific metrics saved to {output_path}")

def visualize_metrics(json_path):
    if not os.path.exists(json_path): return
    with open(json_path, 'r') as f: data = json.load(f)
    os.makedirs("plots", exist_ok=True)
    timesteps = data.get("metadata", {}).get("timesteps", [])

    # Update visualization to use the new nested JSON structure
    all_layers = {**data["Attention"], **data["MLP"]}
    sorted_names = sorted(all_layers.keys(), key=lambda x: [int(s) if s.isdigit() else s for s in x.split('.')])
    
    # ... (rest of plotting code updated for new structure)
    plt.figure(figsize=(14, 8))
    cmap = plt.get_cmap('viridis')
    for i, t in enumerate(timesteps):
        color = cmap(i / len(timesteps))
        t_str = str(t)
        means = np.array([all_layers[n]["fisher_stats"].get(t_str, {}).get("mean", 0) for n in sorted_names])
        stds = np.array([all_layers[n]["fisher_stats"].get(t_str, {}).get("std", 0) for n in sorted_names])
        plt.plot(range(len(sorted_names)), means, color=color, label=f"t={t}", alpha=0.9)
        plt.fill_between(range(len(sorted_names)), means - stds, means + stds, color=color, alpha=0.1)
    
    plt.title("Layer Depth vs Mean Fisher Information (with Std Dev)"); plt.yscale('log'); plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left'); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("plots/fisher_info_stats.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of prompts to profile")
    parser.add_argument("--num_timesteps", type=int, default=10, help="Number of timesteps to profile (uniformly sampled)")
    parser.add_argument("--output", type=str, default="layer_metrics.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    ds = load_dataset("nateraw/parti-prompts", split="train")
    prompts = random.sample(list(ds['Prompt']), args.num_samples)
    timesteps = np.linspace(1000, 0, args.num_timesteps, endpoint=False, dtype=int).tolist()

    print(f"Loading model {args.model_id}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16).to(args.device)
    
    layers = get_linear_layers(pipe.transformer)
    weight_metrics = profile_weights(layers)
    fisher_metrics = calculate_fisher(pipe, layers, args.device, prompts, timesteps)
    
    export_json(weight_metrics, fisher_metrics, args.output, timesteps)
    visualize_metrics(args.output)

if __name__ == "__main__":
    main()

