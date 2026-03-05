#!/usr/bin/env bash
# =============================================================================
# run_mega.sh
# -----------
# LG-CLIP: Evaluate zero-shot classification using generated-image prototypes.
# 
# This is the main evaluation script. It loads pre-extracted features and
# reports the final accuracy. Results are appended to metalog.txt.
#
# Prerequisites:
#   1. Real-image features:  run vanilla_clip.py  (or vanilla_clip_ms.py for --ms1 _ms)
#   2. Gen-image features:   run gen_feat.py       (or muti_scale_gen_feat.py for --ms2 _ms)
#
# Usage:
#   bash scripts/run_mega.sh [OPTIONS]
#
# Options (all optional, defaults shown):
#   --dataset   Dataset name  (default: EUROSAT)
#   --Ngen      N generated images used in features  (default: 5)
#   --LLM       "LLM_" for LLM-guided, "" for plain SD  (default: "")
#   --backbone  CLIP backbone  (default: RN50)
#   --ms1       Real-image feature suffix: "" or "_ms"  (default: "")
#   --ms2       Gen-image feature suffix: "" or "_ms"  (default: "")
#   --image_root Dataset root  (default: ./dataset)
#   --device    cuda:0 | cpu  (default: cuda:0)
#   --seed      Random seed  (default: 2024)
#
# Examples:
#   # Standard (no multi-scale)
#   bash scripts/run_mega.sh --dataset CUB --Ngen 10 --backbone ViT-L/14 --LLM LLM_
#
#   # With multi-scale features
#   bash scripts/run_mega.sh --dataset PET --Ngen 10 --ms1 _ms --ms2 _ms --backbone ViT-L/14
# =============================================================================

set -euo pipefail

DATASET="EUROSAT"
NGEN=5
LLM=""
BACKBONE="RN50"
MS1=""
MS2=""
IMAGE_ROOT="dataset"
DEVICE="cuda:0"
SEED=2024

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)    DATASET="$2";    shift 2 ;;
        --Ngen)       NGEN="$2";       shift 2 ;;
        --LLM)        LLM="$2";        shift 2 ;;
        --backbone)   BACKBONE="$2";   shift 2 ;;
        --ms1)        MS1="$2";        shift 2 ;;
        --ms2)        MS2="$2";        shift 2 ;;
        --image_root) IMAGE_ROOT="$2"; shift 2 ;;
        --device)     DEVICE="$2";     shift 2 ;;
        --seed)       SEED="$2";       shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

echo "========================================================"
echo " Running: mega.py (LG-CLIP prototype-based classification)"
echo "  Dataset  : $DATASET"
echo "  Backbone : $BACKBONE"
echo "  Ngen     : $NGEN"
echo "  LLM      : ${LLM:-'(none)'}"
echo "  ms1      : ${MS1:-'(none)'}"
echo "  ms2      : ${MS2:-'(none)'}"
echo "  Device   : $DEVICE"
echo "========================================================"

python mega.py \
    --dataset    "$DATASET"    \
    --backbone   "$BACKBONE"   \
    --Ngen       "$NGEN"       \
    --LLM        "$LLM"        \
    --ms1        "$MS1"        \
    --ms2        "$MS2"        \
    --image_root "$IMAGE_ROOT" \
    --device     "$DEVICE"     \
    --seed       "$SEED"

echo "Done. Results logged to metalog.txt"
