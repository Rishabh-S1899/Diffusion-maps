#!/bin/bash

# ==============================================================================
# Diffusion-Maps Experiment Runner
# ==============================================================================
# This script runs the three core scenarios for the report comparison.
# Make sure you have generated your optimal_schedule.json first.
# ==============================================================================

# --- CONFIGURATION ---
PROMPTS=10
SCHEDULE="optimal_schedule.json"
SEED=42
BASE_OUT="experiments"

# Check if schedule exists
if [ ! -f "$SCHEDULE" ]; then
    echo "ERROR: Schedule file '$SCHEDULE' not found!"
    echo "Please generate it using: uv run python generate_optimal_schedule.py --data_dir <your_stats_dir>"
    exit 1
fi

echo "===================================================================="
echo "STARTING EXPERIMENTS (Prompts: $PROMPTS, Seed: $SEED)"
echo "===================================================================="

# 1. BASELINE RUN
echo -e "
[1/3] RUNNING BASELINE (No Optimizations)..."
uv run python demo-sd3-5.py 
    --prompts $PROMPTS 
    --seed $SEED 
    --profile 
    --no_cross 
    --output_dir "$BASE_OUT/baseline"

# 2. CACHING RUN
echo -e "
[2/3] RUNNING CACHING (Temporal Output Caching)..."
uv run python demo-sd3-5.py 
    --prompts $PROMPTS 
    --seed $SEED 
    --profile 
    --cache_schedule "$SCHEDULE" 
    --output_dir "$BASE_OUT/cached"

# 3. CACHING + QUANTIZATION RUN
echo -e "
[3/3] RUNNING CACHING + 8-BIT QUANTIZATION..."
uv run python demo-sd3-5.py 
    --prompts $PROMPTS 
    --seed $SEED 
    --profile 
    --cache_schedule "$SCHEDULE" 
    --quantize_cache 
    --output_dir "$BASE_OUT/cached_quantized"

echo -e "
===================================================================="
echo "EXPERIMENTS COMPLETE"
echo "Results are stored in: $BASE_OUT/"
echo "Check your terminal history above for the GFLOPs and VRAM summaries."
echo "===================================================================="
