import torch
from diffusers import StableDiffusion3Pipeline
import os
from attention_map_diffusers import (
    attn_maps,
    init_pipeline,
    save_attention_maps
)
current_directory = os.getcwd()
os.environ['HF_HOME'] = current_directory

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-medium",
    torch_dtype=torch.bfloat16
)
pipe = pipe.to("cuda")

##### 1. Replace modules and Register hook #####
pipe = init_pipeline(pipe)
################################################

# recommend not using batch operations for sd3, as cpu memory could be exceeded.
prompts = [
    # "A photo of a puppy wearing a hat.",
    "A red helicopter flying over a city skyline at sunset.",
]

images = pipe(
    prompts,
    num_inference_steps=15,
    guidance_scale=4.5,
).images

print("This is the generated images:\n", images)
for batch, image in enumerate(images):
    image.save(f'{batch}-sd3-5.png')

##### 2. Process and Save attention map #####

# Save raw attention map tensors for later analysis
import pickle
raw_tensor_dir = 'attn_maps-sd3-5-tensors'
os.makedirs(raw_tensor_dir, exist_ok=True)
for timestep, layers in attn_maps.items():
    timestep_dir = os.path.join(raw_tensor_dir, f'{timestep}')
    os.makedirs(timestep_dir, exist_ok=True)
    for layer, attn_map in layers.items():
        tensor_path = os.path.join(timestep_dir, f'{layer}.pkl')
        with open(tensor_path, 'wb') as f:
            pickle.dump(attn_map.cpu(), f)

# Save visual attention maps as before
save_attention_maps(attn_maps, pipe.tokenizer, prompts, base_dir='attn_maps-sd3-5', unconditional=True)
#############################################