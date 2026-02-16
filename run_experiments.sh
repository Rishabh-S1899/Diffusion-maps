#!/bin/bash

# --- CONFIGURATION ---
PROMPTS=100
SCHEDULE="sample_cache_schedule.json"
SEED=42
BASE_OUT="experiments"

# Check if schedule exists
if [ ! -f "$SCHEDULE" ]; then
    echo "ERROR: Schedule file '$SCHEDULE' not found!"
    exit 1
fi

echo "===================================================================="
echo "STARTING EXPERIMENTS (Prompts: $PROMPTS, Seed: $SEED)"
echo "===================================================================="

# 1. BASELINE RUN
echo -e "\n[1/3] RUNNING BASELINE..."
uv run python demo-sd3-5.py --prompts $PROMPTS --seed $SEED --output_dir "$BASE_OUT/baseline"

# 2. CACHING RUN
echo -e "\n[2/3] RUNNING CACHING..."
uv run python demo-sd3-5.py --prompts $PROMPTS --seed $SEED --cache_schedule "$SCHEDULE" --output_dir "$BASE_OUT/cached"

# 3. CACHING + QUANTIZATION RUN
echo -e "\n[3/3] RUNNING CACHING + 8-BIT QUANTIZATION..."
uv run python demo-sd3-5.py --prompts $PROMPTS --seed $SEED --cache_schedule "$SCHEDULE" --quantize_cache --output_dir "$BASE_OUT/cached_quantized"

echo -e "\n===================================================================="
echo "EXPERIMENTS COMPLETE"
echo "===================================================================="
