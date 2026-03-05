#!/usr/bin/env bash
# =============================================================================
# run_text_gen_filter.sh
# -----------------------
# Filter the top-Ngen generated images per class based on CLIP similarity score.
# Produces a JSON index file used by gen_feat.py when Ngen < 10.
#
# Prerequisite:
#   All 10 generated images per class must already exist (run sd_gen.py / llm_sd_gen.py
#   with --Ngen 10 first).
#
# Usage:
#   bash scripts/run_text_gen_filter.sh [OPTIONS]
#
# Options:
#   --dataset       Dataset name  (default: ImageNet)
#   --Ngen          Number to select per class  (default: 5)
#   --LLM           "LLM_" for LLM-guided, "" for plain SD  (default: LLM_)
#   --backbone      CLIP backbone for scoring  (default: ViT-B/32)
#   --image_root    Dataset root  (default: ./dataset)
#   --gen_root_path Generated image root  (auto-set from LLM flag if empty)
#   --device        cuda:0 | cpu  (default: cuda:0)
#
# Examples:
#   bash scripts/run_text_gen_filter.sh --dataset CUB --Ngen 5 --LLM LLM_
#   bash scripts/run_text_gen_filter.sh --dataset PET --Ngen 3 --backbone ViT-L/14
# =============================================================================

set -euo pipefail

DATASET="ImageNet"
NGEN=5
LLM="LLM_"
BACKBONE="ViT-B/32"
IMAGE_ROOT="dataset"
GEN_ROOT=""
DEVICE="cuda:0"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)       DATASET="$2";    shift 2 ;;
        --Ngen)          NGEN="$2";       shift 2 ;;
        --LLM)           LLM="$2";        shift 2 ;;
        --backbone)      BACKBONE="$2";   shift 2 ;;
        --image_root)    IMAGE_ROOT="$2"; shift 2 ;;
        --gen_root_path) GEN_ROOT="$2";   shift 2 ;;
        --device)        DEVICE="$2";     shift 2 ;;
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

echo "========================================================"
echo " Running: text_gen_Ngen_made.py (CLIP-based image filtering)"
echo "  Dataset  : $DATASET"
echo "  Ngen     : $NGEN (selected from 10)"
echo "  LLM      : ${LLM:-'(none)'}"
echo "  Backbone : $BACKBONE"
echo "  Gen root : $GEN_ROOT"
echo "  Device   : $DEVICE"
echo "========================================================"

python text_gen_Ngen_made.py \
    --dataset       "$DATASET"    \
    --Ngen          "$NGEN"       \
    --LLM           "$LLM"        \
    --backbone      "$BACKBONE"   \
    --image_root    "$IMAGE_ROOT" \
    --gen_root_path "$GEN_ROOT"   \
    --device        "$DEVICE"

echo "Done."
