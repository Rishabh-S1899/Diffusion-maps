import torch
import os
import sys

import random
import re
from datasets import load_dataset
from diffusers import StableDiffusion3Pipeline

# Your local library imports
from attention_map_diffusers import attn_maps, init_pipeline
from attention_map_diffusers.utils import save_attention_stats 

# 1. Load and sample 100 random prompts
ds = load_dataset("nateraw/parti-prompts", split="train")
random_prompts = random.sample(ds['Prompt'], 200)

# 2. Setup Pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")
pipe = init_pipeline(pipe) # Essential for hooking into the attention layers

# 3. Process and Save
for i, prompt in enumerate(random_prompts):
    # Create sanitized folder name
    clean_prompt_snippet = re.sub(r'[^\w\s]', '', prompt[:20]).strip().replace(' ', '_')
    output_dir = f"outputs/{clean_prompt_snippet}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run inference
    # The 'init_pipeline' hooks will populate 'attn_maps' during this call
    image = pipe(prompt, num_inference_steps=15, guidance_scale=4.5).images[0]
    
    # Save image
    image.save(os.path.join(output_dir, "result.png"))

    # SAVE STATISTICS FOR THIS SPECIFIC PROMPT
    # We save into a 'stats' subfolder inside the prompt's directory
    stats_path = os.path.join(output_dir, 'attn_stats')
    save_attention_stats(attn_maps, base_dir=stats_path)
    
    # IMPORTANT: If your library appends to attn_maps, 
    # you may need to clear attn_maps here so the next prompt starts fresh.
    # Check if your library has a clear() method or re-initialize it.
    attn_maps.clear() 

print("Processing complete with statistics saved for each prompt.")