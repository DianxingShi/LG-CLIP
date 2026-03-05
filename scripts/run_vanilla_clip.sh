#!/usr/bin/env bash
# =============================================================================
# run_vanilla_clip.sh
# ---------------------
# Evaluate zero-shot CLIP accuracy on REAL images.
# Extracts and caches CLIP visual + text features, then reports accuracy.
#
# Usage:
#   bash scripts/run_vanilla_clip.sh [OPTIONS]
#
# Options (all optional, defaults shown):
#   --dataset    CUB | FLO | PET | FOOD | ImageNet | EUROSAT  (default: FLO)
#   --backbone   RN50 | RN101 | ViT-B/32 | ViT-B/16 | ViT-L/14  (default: RN50)
#   --image_root Path to dataset root directory  (default: ./dataset)
#   --device     cuda:0 | cuda:1 | cpu  (default: cuda:0)
#   --seed       Random seed  (default: 2024)
#
# Examples:
#   bash scripts/run_vanilla_clip.sh
#   bash scripts/run_vanilla_clip.sh --dataset CUB --backbone ViT-L/14
#   bash scripts/run_vanilla_clip.sh --dataset ImageNet --device cuda:1
# =============================================================================

set -euo pipefail

# ------------ Default values ------------
DATASET="FLO"
BACKBONE="RN50"
IMAGE_ROOT="dataset"
DEVICE="cuda:0"
SEED=2024

# ------------ Parse arguments ------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)    DATASET="$2";    shift 2 ;;
        --backbone)   BACKBONE="$2";   shift 2 ;;
        --image_root) IMAGE_ROOT="$2"; shift 2 ;;
        --device)     DEVICE="$2";     shift 2 ;;
        --seed)       SEED="$2";       shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

echo "========================================================"
echo " Running: vanilla_clip.py (real image features)"
echo "  Dataset   : $DATASET"
echo "  Backbone  : $BACKBONE"
echo "  Image root: $IMAGE_ROOT"
echo "  Device    : $DEVICE"
echo "  Seed      : $SEED"
echo "========================================================"

python vanilla_clip.py \
    --dataset    "$DATASET"    \
    --backbone   "$BACKBONE"   \
    --image_root "$IMAGE_ROOT" \
    --device     "$DEVICE"     \
    --seed       "$SEED"

echo "Done."
