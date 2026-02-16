import torch
import os
import argparse
import json
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

def calculate_metrics(base_dir, use_clip=True, use_reward=True):
    print(f"Calculating metrics for: {base_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Load Models ---
    clip_model, clip_processor = None, None
    reward_model = None
    
    if use_clip:
        print("Loading CLIP model (openai/clip-vit-base-patch32)...")
        clip_id = "openai/clip-vit-base-patch32"
        clip_model = CLIPModel.from_pretrained(clip_id).to(device)
        clip_processor = CLIPProcessor.from_pretrained(clip_id)
        
    if use_reward:
        print("Loading ImageReward model (v1.0)...")
        try:
            import ImageReward as RM
            # RM.load handles downloading and loading the weights
            reward_model = RM.load("ImageReward-v1.0")
            reward_model.to(device)
            reward_model.eval()
        except Exception as e:
            print(f"\nCRITICAL ERROR loading ImageReward: {e}")
            print("Please ensure you ran: uv pip install image-reward")
            use_reward = False

    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist.")
        return None

    clip_scores = []
    reward_scores = []
    
    # Identify prompt folders
    prompt_folders = [f for f in base_path.iterdir() if f.is_dir()]
    
    for folder in tqdm(prompt_folders, desc="Processing Images"):
        img_path = folder / "result.png"
        txt_path = folder / "prompt.txt"
        
        if not img_path.exists() or not txt_path.exists():
            continue
            
        try:
            image = Image.open(img_path).convert("RGB")
            with open(txt_path, 'r') as f:
                prompt = f.read().strip()
            
            # 1. Calculate CLIP
            if use_clip:
                inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    # logits_per_image is cosine similarity * 100
                    score = clip_model(**inputs).logits_per_image.item() / 100.0
                    clip_scores.append(score)
            
            # 2. Calculate ImageReward
            if use_reward:
                with torch.no_grad():
                    # The model.score method expects (prompt, list_of_images)
                    # It returns a list of floats
                    score = reward_model.score(prompt, [image])
                    if isinstance(score, list):
                        score = score[0]
                    reward_scores.append(float(score))
                    
        except Exception as e:
            print(f"Error processing {folder.name}: {e}")

    # --- Summary ---
    results = {"directory": base_dir, "sample_count": len(prompt_folders)}
    
    print(f"\nFinal Results for {base_dir}:")
    if use_clip and clip_scores:
        avg_clip = sum(clip_scores) / len(clip_scores)
        results["average_clip_score"] = avg_clip
        print(f"  Avg CLIP Score:   {avg_clip:.4f}")
        
    if use_reward and reward_scores:
        avg_reward = sum(reward_scores) / len(reward_scores)
        results["average_reward_score"] = avg_reward
        print(f"  Avg ImageReward:  {avg_reward:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Directory containing prompt subfolders")
    parser.add_argument("--clip", action="store_true", help="Enable CLIP calculation")
    parser.add_argument("--reward", action="store_true", help="Enable ImageReward calculation")
    parser.add_argument("--output_json", type=str, default=None, help="Save result to a JSON file")
    args = parser.parse_args()
    
    if not args.clip and not args.reward:
        print("Warning: No metrics enabled. Use --clip or --reward.")
        return

    result = calculate_metrics(args.dir, use_clip=args.clip, use_reward=args.reward)
    
    if result and args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output_json}")

if __name__ == "__main__":
    main()
