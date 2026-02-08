import torch
import os
import sys
import random
import re
from datasets import load_dataset
from diffusers import StableDiffusion3Pipeline

# ... (Keep your sys.path and environment setup from previous steps) ...

# 1. Load and sample 100 random prompts
ds = load_dataset("nateraw/parti-prompts", split="train")
all_prompts = ds['Prompt']
random_prompts = random.sample(all_prompts, 500)

# 2. Setup Pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")
# pipe = init_pipeline(pipe) # Ensure this is defined/imported from your attention_map_diffusers

# 3. Process and Save
for i, prompt in enumerate(random_prompts):
    # Create folder name: first 20 chars, removing non-alphanumeric chars
    clean_prompt_snippet = re.sub(r'[^\w\s]', '', prompt[:20]).strip().replace(' ', '_')
    output_dir = f"outputs/{clean_prompt_snippet}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Run inference (individual processing to avoid OOM)
    image = pipe(prompt, num_inference_steps=15, guidance_scale=4.5).images[0]
    
    # Save image and a text file with the full prompt for reference
    image.save(os.path.join(output_dir, "result.png"))
    with open(os.path.join(output_dir, "prompt.txt"), "w") as f:
        f.write(prompt)

    # If your attention_map_diffusers logic saves per-run stats:
    # save_attention_stats(attn_maps, base_dir=os.path.join(output_dir, 'stats'))

print("Processing complete.")