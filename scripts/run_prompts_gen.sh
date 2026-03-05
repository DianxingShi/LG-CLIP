#!/usr/bin/env bash
# =============================================================================
# run_prompts_gen.sh
# ------------------
# Generate LLM-enriched image descriptions (prompts) for each class
# via the xAI Grok API. Output is saved as a JSON file used by llm_sd_gen.py.
#
# Usage:
#   bash scripts/run_prompts_gen.sh [OPTIONS]
#
# Options (all optional, defaults shown):
#   --dataset     Dataset name  (default: CUB)
#   --image_root  Dataset root  (default: ./dataset)
#   --output_dir  Where to save prompts JSON  (default: ./dataset/prompts)
#   --api_key     Your xAI API key  (REQUIRED)
#   --num_prompts Prompts per class  (default: 10)
#
# Examples:
#   bash scripts/run_prompts_gen.sh --dataset CUB --api_key xai-xxxxxxxx
#   bash scripts/run_prompts_gen.sh --dataset PET --num_prompts 5 --api_key xai-xxxxxxxx
# =============================================================================

set -euo pipefail

DATASET="CUB"
IMAGE_ROOT="dataset"
OUTPUT_DIR="dataset/prompts"
API_KEY=""
NUM_PROMPTS=10

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)     DATASET="$2";     shift 2 ;;
        --image_root)  IMAGE_ROOT="$2";  shift 2 ;;
        --output_dir)  OUTPUT_DIR="$2";  shift 2 ;;
        --api_key)     API_KEY="$2";     shift 2 ;;
        --num_prompts) NUM_PROMPTS="$2"; shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

# Validate that API key was provided
if [[ -z "$API_KEY" ]]; then
    echo "[ERROR] --api_key is required. Get your key from https://x.ai/"
    exit 1
fi

echo "========================================================"
echo " Running: prompts_gen.py (LLM prompt generation)"
echo "  Dataset     : $DATASET"
echo "  Output dir  : $OUTPUT_DIR"
echo "  Num prompts : $NUM_PROMPTS"
echo "========================================================"

python prompts_gen.py \
    --dataset     "$DATASET"     \
    --image_root  "$IMAGE_ROOT"  \
    --output_dir  "$OUTPUT_DIR"  \
    --api_key     "$API_KEY"     \
    --num_prompts "$NUM_PROMPTS"

echo "Done. Prompts saved to: $OUTPUT_DIR/prompts_for_${DATASET}.json"
