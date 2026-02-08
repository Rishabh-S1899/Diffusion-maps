import torch
import os
import re
import json
from datasets import load_dataset
from diffusers import StableDiffusion3Pipeline
from attention_map_diffusers import attn_maps, init_pipeline
from attention_map_diffusers.utils import save_attention_stats

# --- Setup ---
base_url = "https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_{i:06d}.tar"
urls = [base_url.format(i=i) for i in range(46)]

# 1. Load 500 random points via streaming
dataset = load_dataset("webdataset", data_files={"train": urls}, split="train", streaming=True)
dataset = dataset.shuffle(seed=42, buffer_size=1000).take(500)

# 2. Pipeline setup
pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.bfloat16
).to("cuda")
pipe = init_pipeline(pipe)

print("Starting inference...")

# 3. Execution Loop
for i, sample in enumerate(dataset):
    # --- PROMPT EXTRACTION LOGIC ---
    prompt = ""
    
    # Priority 1: Check for .txt file content (most common in webdataset)
    if "txt" in sample:
        raw_txt = sample["txt"]
        prompt = raw_txt.decode('utf-8') if isinstance(raw_txt, bytes) else raw_txt
    
    # Priority 2: Check for .json file content (used in some shards of this dataset)
    elif "json" in sample:
        raw_json = sample["json"]
        # Decode bytes if necessary, then parse JSON to find the 'prompt' or 'caption' key
        json_data = json.loads(raw_json.decode('utf-8')) if isinstance(raw_json, bytes) else raw_json
        prompt = json_data.get("prompt", json_data.get("caption", ""))

    # print(f"prompt is {prompt}")
    # If both fail, skip to avoid generating for an empty string
    if not prompt or prompt.strip() == "":
        print(f"Skipping sample {i}: No prompt found.")
        continue

    # --- SAVE & INFERENCE ---
    clean_prompt = re.sub(r'[^\w\s]', '', prompt[:20]).strip().replace(' ', '_')
    output_dir = f"outputs_jacky/{i}_{clean_prompt}"
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        image = pipe(prompt, num_inference_steps=15, guidance_scale=4.5).images[0]
    
    image.save(os.path.join(output_dir, "result.png"))
    save_attention_stats(attn_maps, base_dir=os.path.join(output_dir, 'attn_stats'))
    
    attn_maps.clear()
    torch.cuda.empty_cache()

    if i % 10 == 0:
        print(f"Progress: {i}/500 | Current Prompt: {prompt[:40]}...")