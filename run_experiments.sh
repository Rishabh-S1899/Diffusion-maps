#!/bin/bash

# --- CONFIGURATION ---
PROMPTS=2
SCHEDULE="protected_cache_schedule.json" # Your working Sandwich schedule
SEED=42
BASE_OUT="experiments"

# Check if schedule exists
if [ ! -f "$SCHEDULE" ]; then
    echo "ERROR: Schedule file '$SCHEDULE' not found! Please run the schedule generator first."
    exit 1
fi

echo "===================================================================="
echo "STARTING LARGE-SCALE VALIDATION (Prompts: $PROMPTS, Seed: $SEED)"
echo "Architecture: Protected Boundary Caching + True 4-bit/8-bit Packing"
echo "===================================================================="

# 1. BASELINE RUN 
# Expectation: ~16.76 GB VRAM, ~339k GFLOPs, Perfect Quality
echo -e "\n[1/3] RUNNING BASELINE (Full 16-bit Compute)..."
uv run python demo-sd3-5.py --prompts $PROMPTS --seed $SEED --output_dir "$BASE_OUT/baseline" --profile

# 2. CACHING ONLY RUN 
# Expectation: ~16.76 GB VRAM, ~280k GFLOPs, Perfect Quality
echo -e "\n[2/3] RUNNING CACHING ONLY (Protected Schedule, 16-bit Storage)..."
uv run python demo-sd3-5.py --prompts $PROMPTS --seed $SEED --cache_schedule "$SCHEDULE" --output_dir "$BASE_OUT/cached" --profile

# 3. CACHING + QUANTIZATION RUN
# Expectation: <16.00 GB VRAM, ~280k GFLOPs, Perfect Quality
echo -e "\n[3/3] RUNNING CACHING + TRUE 4/8-BIT QUANTIZATION..."
uv run python demo-sd3-5.py --prompts $PROMPTS --seed $SEED --cache_schedule "$SCHEDULE" --quantize_cache --output_dir "$BASE_OUT/cached_quantized" --profile

echo -e "\n===================================================================="
echo "EXPERIMENTS COMPLETE! Output saved to: $BASE_OUT"
echo "===================================================================="