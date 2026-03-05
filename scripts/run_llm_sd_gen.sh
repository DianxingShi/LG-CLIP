#!/usr/bin/env bash
# =============================================================================
# run_llm_sd_gen.sh
# -----------------
# Generate images using Stable Diffusion guided by LLM-enriched prompts.
# Prompts are loaded from a JSON file produced by run_prompts_gen.sh.
#
# Usage:
#   bash scripts/run_llm_sd_gen.sh [OPTIONS]
#
# Options (all optional, defaults shown):
#   --dataset       Dataset name  (default: CUB)
#   --json_file     Path to prompts JSON  (default: ./dataset/prompts/prompts_for_CUB.json)
#   --Ngen          Images per class  (default: 10)
#   --gen_root_path Output directory  (default: ./dataset/LLM_SD_gen)
#   --image_root    Dataset root  (default: ./dataset)
#   --sd_version    SD version: 2.1 | xl | 1.4  (default: 2.1)
#   --device        cuda:0 | cpu  (default: cuda:0)
#   --seed          Random seed  (default: 2024)
#
# Examples:
#   bash scripts/run_llm_sd_gen.sh --dataset CUB
#   bash scripts/run_llm_sd_gen.sh --dataset FLO --json_file dataset/prompts/prompts_for_FLO.json
# =============================================================================

set -euo pipefail

DATASET="CUB"
JSON_FILE=""   # will be auto-set if not provided
NGEN=10
GEN_ROOT="dataset/LLM_SD_gen"
IMAGE_ROOT="dataset"
SD_VERSION="2.1"
DEVICE="cuda:0"
SEED=2024

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)       DATASET="$2";    shift 2 ;;
        --json_file)     JSON_FILE="$2";  shift 2 ;;
        --Ngen)          NGEN="$2";       shift 2 ;;
        --gen_root_path) GEN_ROOT="$2";   shift 2 ;;
        --image_root)    IMAGE_ROOT="$2"; shift 2 ;;
        --sd_version)    SD_VERSION="$2"; shift 2 ;;
        --device)        DEVICE="$2";     shift 2 ;;
        --seed)          SEED="$2";       shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

# Auto-set json_file based on dataset if not provided
if [[ -z "$JSON_FILE" ]]; then
    JSON_FILE="dataset/prompts/prompts_for_${DATASET}.json"
fi

# Validate that prompts JSON exists
if [[ ! -f "$JSON_FILE" ]]; then
    echo "[ERROR] Prompts JSON not found: $JSON_FILE"
    echo "  Run: bash scripts/run_prompts_gen.sh --dataset $DATASET --api_key YOUR_KEY"
    exit 1
fi

# Map sd_version to Python flag
case "$SD_VERSION" in
    "2.1") SD_FLAG="--sd_2_1" ;;
    "xl")  SD_FLAG="--sd_xl"  ;;
    "1.4") SD_FLAG=""         ;;
    *) echo "[ERROR] Unknown sd_version: $SD_VERSION (use 2.1 | xl | 1.4)"; exit 1 ;;
esac

echo "========================================================"
echo " Running: llm_sd_gen.py (LLM-guided SD image generation)"
echo "  Dataset   : $DATASET"
echo "  JSON file : $JSON_FILE"
echo "  Ngen      : $NGEN"
echo "  SD version: $SD_VERSION"
echo "  Gen root  : $GEN_ROOT"
echo "  Device    : $DEVICE"
echo "========================================================"

python llm_sd_gen.py \
    --dataset       "$DATASET"    \
    --json_file     "$JSON_FILE"  \
    --Ngen          "$NGEN"       \
    --gen_root_path "$GEN_ROOT"   \
    --image_root    "$IMAGE_ROOT" \
    $SD_FLAG                      \
    --device        "$DEVICE"     \
    --seed          "$SEED"

echo "Done."
