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
    # Attention related
    if any(x in name_lower for x in ['to_q', 'to_k', 'to_v', 'to_out', 'add_q_proj', 'add_k_proj', 'add_v_proj', 'to_add_out', 'attn']):
        return "Attention"
    # MLP related
    if any(x in name_lower for x in ['ff', 'feed_forward', 'proj_out', 'proj_in']):
        return "MLP"
    return "Other"

def get_linear_layers(model):
    """Find all nn.Linear layers and categorize them."""
    layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            category = categorize_layer(name)
            layers.append({
                'name': name,
                'module': module,
                'category': category
            })
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
            # We use CPU for SVD if the weight is too large for GPU memory, 
            # but usually for profiling we have enough VRAM.
            singular_values = torch.linalg.svdvals(weight)
            s_max = singular_values[0].item()
            s_min = singular_values[-1].item()
            condition_number = s_max / (s_min + 1e-10)
            
            metrics[name] = {
                'category': layer_info['category'],
                'singular_values': singular_values.cpu().numpy().tolist(),
                'spectral_norm': s_max,
                'condition_number': condition_number
            }
        except Exception as e:
            print(f"Error profiling layer {name}: {e}")
            
    return metrics

def calculate_fisher(pipe, layers, device, prompt=None, timesteps=[900, 500, 100]):
    """Calculate Empirical Diagonal Fisher Information using real or dummy encodings."""
    print(f"Calculating Fisher Information for timesteps {timesteps}...")
    
    model = pipe.transformer
    model.train() # Enable gradient tracking
    
    # 1. Prepare Encodings
    batch_size = 1
    if prompt:
        print(f"  Encoding prompt: {prompt[:50]}...")
        with torch.no_grad():
            (
                prompt_embeds,
                negative_prompt_embeds,
                pooled_prompt_embeds,
                negative_pooled_prompt_embeds,
            ) = pipe.encode_prompt(prompt=prompt, prompt_2=None, prompt_3=None)
            
            # SD3 uses prompt_embeds and pooled_prompt_embeds
            encoder_hidden_states = prompt_embeds.to(device=device, dtype=torch.bfloat16)
            pooled_projections = pooled_prompt_embeds.to(device=device, dtype=torch.bfloat16)
    else:
        print("  Using dummy encodings...")
        encoder_hidden_states = torch.randn(batch_size, 154, 4096, device=device, dtype=torch.bfloat16)
        pooled_projections = torch.randn(batch_size, 2048, device=device, dtype=torch.bfloat16)
    
    hidden_states = torch.randn(batch_size, 16, 64, 64, device=device, dtype=torch.bfloat16)
    
    fisher_metrics = {}
    
    # We only care about transformer gradients
    for param in model.parameters():
        param.requires_grad = True

    for t_val in timesteps:
        print(f"  Processing timestep t={t_val}...")
        timestep = torch.tensor([t_val], device=device, dtype=torch.bfloat16)
        
        model.zero_grad()
        
        # Forward pass
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            return_dict=True
        ).sample
        
        # Empirical Fisher: mean squared gradient
        # We use a simple sum of squares as a proxy for the likelihood gradient
        loss = output.pow(2).mean()
        loss.backward()
        
        for layer_info in layers:
            name = layer_info['name']
            module = layer_info['module']
            
            if module.weight.grad is not None:
                fisher = module.weight.grad.pow(2).mean().item()
            else:
                fisher = 0.0
            
            if name not in fisher_metrics:
                fisher_metrics[name] = {}
            fisher_metrics[name][f"fisher_t{t_val}"] = fisher
            
        # Clean up aggressively
        model.zero_grad()
        torch.cuda.empty_cache()
        
    return fisher_metrics

def export_json(weight_metrics, fisher_metrics, output_path="layer_metrics.json"):
    """Merge and export metrics to a structured JSON."""
    combined = {"Attention": {}, "MLP": {}, "Other": {}}
    all_names = set(weight_metrics.keys()) | set(fisher_metrics.keys())
    
    for name in all_names:
        w_m = weight_metrics.get(name, {})
        f_m = fisher_metrics.get(name, {})
        category = w_m.get('category', 'Other')
        
        entry = {"name": name, **w_m, **f_m}
        combined[category][name] = entry
        
    with open(output_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"Metrics saved to {output_path}")

def visualize_metrics(json_path="layer_metrics.json"):
    """Generate plots from the exported JSON."""
    if not os.path.exists(json_path): return
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    os.makedirs("plots", exist_ok=True)
    
    # 1. Scree Plot
    plt.figure(figsize=(10, 6))
    # Sample a few layers for visualization
    for cat in ["Attention", "MLP"]:
        layers = list(data[cat].keys())
        if layers:
            # Pick middle layer for representative plot
            name = layers[len(layers)//2]
            sv = data[cat][name]["singular_values"]
            plt.plot(sv[:100], label=f"{cat}: {name.split('.')[-1]}")
        
    plt.title("Scree Plot (First 100 Singular Values)")
    plt.xlabel("Index")
    plt.ylabel("Singular Value")
    plt.legend()
    plt.yscale('log')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig("plots/scree_plot.png")
    
    # 2. Layer Depth vs Condition Number
    plt.figure(figsize=(12, 6))
    for cat, color in [("Attention", "blue"), ("MLP", "red")]:
        layers = data[cat]
        if not layers: continue
        names = sorted(layers.keys(), key=lambda x: [int(s) if s.isdigit() else s for s in x.split('.')])
        values = [layers[n]["condition_number"] for n in names]
        plt.plot(range(len(names)), values, marker='o', markersize=3, color=color, label=cat)

    plt.title("Layer Depth vs Condition Number")
    plt.xlabel("Layer Sequence")
    plt.ylabel("Condition Number")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("plots/condition_number.png")
    
    # 3. Layer Depth vs Fisher Information
    plt.figure(figsize=(12, 6))
    timesteps = [900, 500, 100]
    colors = ['green', 'orange', 'purple']
    
    # Combine Attention and MLP for a continuous depth view
    all_layers = {**data["Attention"], **data["MLP"]}
    sorted_names = sorted(all_layers.keys(), key=lambda x: [int(s) if s.isdigit() else s for s in x.split('.')])
    
    for t, color in zip(timesteps, colors):
        metric = f"fisher_t{t}"
        values = [all_layers[n].get(metric, 0) for n in sorted_names]
        plt.plot(range(len(sorted_names)), values, color=color, label=f"t={t}", alpha=0.8)
        
    plt.title("Layer Depth vs Mean Fisher Information")
    plt.xlabel("Layer Sequence")
    plt.ylabel("Fisher Information")
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("plots/fisher_info.png")
    
    print("Plots generated successfully in 'plots/' folder.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-3.5-medium")
    parser.add_argument("--prompt", type=str, default=None, help="Specific prompt to use for profiling")
    parser.add_argument("--use_dataset", action="store_true", help="Use a random prompt from parti-prompts")
    parser.add_argument("--output", type=str, default="layer_metrics.json")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    
    # 1. Load Dataset if requested
    selected_prompt = args.prompt
    if args.use_dataset and not selected_prompt:
        print("Loading parti-prompts dataset...")
        ds = load_dataset("nateraw/parti-prompts", split="train")
        selected_prompt = random.choice(ds['Prompt'])
        print(f"Selected random prompt: {selected_prompt}")

    # 2. Load Model
    print(f"Loading model {args.model_id}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16
    ).to(args.device)
    
    # 3. Profile
    layers = get_linear_layers(pipe.transformer)
    print(f"Found {len(layers)} linear layers.")
    
    weight_metrics = profile_weights(layers)
    fisher_metrics = calculate_fisher(pipe, layers, args.device, prompt=selected_prompt)
    
    # 4. Export & Plot
    export_json(weight_metrics, fisher_metrics, args.output)
    visualize_metrics(args.output)

if __name__ == "__main__":
    main()

