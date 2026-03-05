#!/usr/bin/env bash
# =============================================================================
# run_gen_feat.sh
# ---------------
# Extract CLIP features from generated images and evaluate generation quality
# via zero-shot classification accuracy.
#
# Prerequisite:
#   Generated images must exist in <gen_root_path> (from sd_gen.py or llm_sd_gen.py).
#
# Usage:
#   bash scripts/run_gen_feat.sh [OPTIONS]
#
# Options (all optional, defaults shown):
#   --dataset       Dataset name  (default: ImageNet)
#   --Ngen          Images per class  (default: 10)
#   --LLM           "LLM_" for LLM-guided, "" for plain SD  (default: "")
#   --backbone      CLIP backbone  (default: RN50)
#   --image_root    Dataset root  (default: ./dataset)
#   --gen_root_path Generated image root  (default: auto-set from LLM flag)
#   --sd_version    SD version: 2.1 | xl | 1.4  (default: 2.1)
#   --device        cuda:0 | cpu  (default: cuda:0)
#   --seed          Random seed  (default: 2024)
#
# Examples:
#   # Plain SD features
#   bash scripts/run_gen_feat.sh --dataset CUB --backbone ViT-L/14
#   # LLM+SD features
#   bash scripts/run_gen_feat.sh --dataset CUB --LLM LLM_ --backbone ViT-L/14
# =============================================================================

set -euo pipefail

DATASET="ImageNet"
NGEN=10
LLM=""
BACKBONE="RN50"
IMAGE_ROOT="dataset"
GEN_ROOT=""   # auto-set below
SD_VERSION="2.1"
DEVICE="cuda:0"
SEED=2024

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)       DATASET="$2";    shift 2 ;;
        --Ngen)          NGEN="$2";       shift 2 ;;
        --LLM)           LLM="$2";        shift 2 ;;
        --backbone)      BACKBONE="$2";   shift 2 ;;
        --image_root)    IMAGE_ROOT="$2"; shift 2 ;;
        --gen_root_path) GEN_ROOT="$2";   shift 2 ;;
        --sd_version)    SD_VERSION="$2"; shift 2 ;;
        --device)        DEVICE="$2";     shift 2 ;;
        --seed)          SEED="$2";       shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

# Auto-set gen_root_path based on LLM flag
if [[ -z "$GEN_ROOT" ]]; then
    if [[ "$LLM" == "LLM_" ]]; then
        GEN_ROOT="dataset/LLM_SD_gen"
    else
        GEN_ROOT="dataset/SD_gen"
    fi
fi

# Map sd_version
case "$SD_VERSION" in
    "2.1") SD_FLAG="--sd_2_1" ;;
    "xl")  SD_FLAG="--sd_xl"  ;;
    "1.4") SD_FLAG=""         ;;
    *) echo "[ERROR] Unknown sd_version: $SD_VERSION"; exit 1 ;;
esac

echo "========================================================"
echo " Running: gen_feat.py (generated image feature extraction)"
echo "  Dataset   : $DATASET"
echo "  Backbone  : $BACKBONE"
echo "  Ngen      : $NGEN"
echo "  LLM       : ${LLM:-'(none)'}"
echo "  SD version: $SD_VERSION"
echo "  Gen root  : $GEN_ROOT"
echo "  Device    : $DEVICE"
echo "========================================================"

python gen_feat.py \
    --dataset       "$DATASET"    \
    --backbone      "$BACKBONE"   \
    --Ngen          "$NGEN"       \
    --LLM           "$LLM"        \
    --image_root    "$IMAGE_ROOT" \
    --gen_root_path "$GEN_ROOT"   \
    $SD_FLAG                      \
    --device        "$DEVICE"     \
    --seed          "$SEED"

echo "Done."
