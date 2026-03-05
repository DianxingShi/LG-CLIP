#!/usr/bin/env bash
# =============================================================================
# run_sd_gen.sh
# -------------
# Generate images with Stable Diffusion using simple class-name prompts.
# Prompts format: "A photo of a <classname>."
#
# Usage:
#   bash scripts/run_sd_gen.sh [OPTIONS]
#
# Options (all optional, defaults shown):
#   --dataset       Dataset name  (default: ImageNet)
#   --Ngen          Images per class  (default: 10)
#   --gen_root_path Output directory  (default: ./dataset/SD_gen)
#   --image_root    Dataset root  (default: ./dataset)
#   --sd_version    SD version: 2.1 | xl | 1.4  (default: 2.1)
#   --device        cuda:0 | cpu  (default: cuda:0)
#   --seed          Random seed  (default: 2024)
#
# Examples:
#   bash scripts/run_sd_gen.sh --dataset CUB --Ngen 10
#   bash scripts/run_sd_gen.sh --dataset FLO --sd_version xl
# =============================================================================

set -euo pipefail

DATASET="ImageNet"
NGEN=10
GEN_ROOT="dataset/SD_gen"
IMAGE_ROOT="dataset"
SD_VERSION="2.1"    # one of: 2.1, xl, 1.4
DEVICE="cuda:0"
SEED=2024

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)       DATASET="$2";     shift 2 ;;
        --Ngen)          NGEN="$2";        shift 2 ;;
        --gen_root_path) GEN_ROOT="$2";    shift 2 ;;
        --image_root)    IMAGE_ROOT="$2";  shift 2 ;;
        --sd_version)    SD_VERSION="$2";  shift 2 ;;
        --device)        DEVICE="$2";      shift 2 ;;
        --seed)          SEED="$2";        shift 2 ;;
        *) echo "[ERROR] Unknown argument: $1"; exit 1 ;;
    esac
done

# Translate --sd_version into the appropriate flag for sd_gen.py
case "$SD_VERSION" in
    "2.1") SD_FLAG="--sd_2_1" ;;
    "xl")  SD_FLAG="--sd_xl"  ;;
    "1.4") SD_FLAG=""         ;;  # default fallback in sd_gen.py
    *) echo "[ERROR] Unknown sd_version: $SD_VERSION (use 2.1 | xl | 1.4)"; exit 1 ;;
esac

echo "========================================================"
echo " Running: sd_gen.py (plain SD image generation)"
echo "  Dataset   : $DATASET"
echo "  Ngen      : $NGEN"
echo "  SD version: $SD_VERSION"
echo "  Gen root  : $GEN_ROOT"
echo "  Device    : $DEVICE"
echo "========================================================"

python sd_gen.py \
    --dataset       "$DATASET"    \
    --Ngen          "$NGEN"       \
    --gen_root_path "$GEN_ROOT"   \
    --image_root    "$IMAGE_ROOT" \
    $SD_FLAG                      \
    --device        "$DEVICE"     \
    --seed          "$SEED"

echo "Done."
