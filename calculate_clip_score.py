import torch
import os
import argparse
import json
from pathlib import Path
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

def calculate_clip_scores(base_dir):
    print(f"Calculating CLIP scores for: {base_dir}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-base-patch32"
    
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"Error: Directory {base_dir} does not exist.")
        return None

    scores = []
    
    # Each sub-folder is a prompt snippet
    prompt_folders = [f for f in base_path.iterdir() if f.is_dir()]
    
    for folder in tqdm(prompt_folders, desc="Processing Images"):
        img_path = folder / "result.png"
        txt_path = folder / "prompt.txt"
        
        if not img_path.exists() or not txt_path.exists():
            continue
            
        try:
            # Load Image and Prompt
            image = Image.open(img_path).convert("RGB")
            with open(txt_path, 'r') as f:
                prompt = f.read().strip()
            
            # Process
            inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                # CLIP Score is typically the cosine similarity * 100
                # We divide by 100 here to get a 0-1 range for the average
                score = outputs.logits_per_image.item() / 100.0
                scores.append(score)
        except Exception as e:
            print(f"Error processing {folder.name}: {e}")

    if not scores:
        print("No valid image/prompt pairs found.")
        return None
        
    avg_score = sum(scores) / len(scores)
    print(f"Results for {base_dir}:")
    print(f"  Count: {len(scores)}")
    print(f"  Average CLIP Score: {avg_score:.4f}")
    
    return {
        "directory": base_dir,
        "average_clip_score": avg_score,
        "sample_count": len(scores)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True, help="Directory containing prompt subfolders")
    parser.add_argument("--output_json", type=str, default=None, help="Save result to a JSON file")
    args = parser.parse_args()
    
    result = calculate_clip_scores(args.dir)
    
    if result and args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output_json}")

if __name__ == "__main__":
    main()
