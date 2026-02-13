import torch
import os
import sys
import random
import re
from datasets import load_dataset
from diffusers import StableDiffusion3Pipeline

from attention_map_diffusers import attn_maps, init_pipeline
from attention_map_diffusers.utils import save_attention_stats 

# 1. Load and sample 200 random prompts
ds = load_dataset("nateraw/parti-prompts", split="train")
random_prompts = random.sample(ds['Prompt'], 500)

# 2. Setup Pipeline
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")
pipe = init_pipeline(pipe)

# 3. Process and Save
for i, prompt in enumerate(random_prompts):
    print(f"Processing prompt {i+1}/500")
    clean_prompt_snippet = re.sub(r'[^\w\s]', '', prompt[:20]).strip().replace(' ', '_')
    output_dir = f"outputs_parti-prompts/{clean_prompt_snippet}"
    
    os.makedirs(output_dir, exist_ok=True)

    ###############################################################
    # CRITICAL: Clear prev_attn_map before each new prompt
    for name, module in pipe.transformer.named_modules():
        if hasattr(module, 'processor'):
            if hasattr(module.processor, 'prev_attn_map'):
                delattr(module.processor, 'prev_attn_map')
            if hasattr(module.processor, 'prev_self_attn_map'):
                delattr(module.processor, 'prev_self_attn_map')
    ###############################################################

    # Run inference
    image = pipe(prompt, num_inference_steps=15, guidance_scale=4.5).images[0]
    
    # Save image
    image.save(os.path.join(output_dir, "result.png"))
    with open(os.path.join(output_dir, "prompt.txt"), 'w') as f:
        f.write(prompt)

    save_attention_stats(attn_maps, base_dir=output_dir)
    
    # Clear for next prompt
    attn_maps.clear()

print("Processing complete with statistics saved for each prompt.")