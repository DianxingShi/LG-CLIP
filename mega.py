"""
mega.py
-------
LG-CLIP: Prototype-based Zero-shot Classification with Generated Images.

This script:
  1. Loads pre-extracted CLIP features for real test images (from vanilla_clip.py).
  2. Loads pre-extracted CLIP features for generated images (from gen_feat.py).
  3. Computes class prototypes by averaging generated-image embeddings per class.
  4. Evaluates zero-shot classification accuracy using cosine similarity between
     real test features and the generated-image prototypes.
  5. Logs results to metalog.txt for experiment tracking.

Usage:
  python mega.py --dataset PET --backbone ViT-L/14 --Ngen 10 --LLM LLM_
"""

import os
import h5py
import torch
import random
import argparse
import numpy as np
import torch.backends.cudnn as cudnn

import clip
from utils.myDataset import (
    CUBDataset, FLODataset, PETDataset, FOODDataset,
    ImageNetDataset, EUROSATDataset
)

# ===========================================================================
# Argument Parsing
# ===========================================================================

projectPath = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(
    description="LG-CLIP: Zero-shot classification using generated-image prototypes."
)
# --- Paths ---
parser.add_argument('--dataset',    default='EUROSAT',
                    help='Dataset name: CUB | FLO | PET | FOOD | ImageNet | EUROSAT')
parser.add_argument('--image_root', default=os.path.join(projectPath, 'dataset'),
                    help='Root directory containing all datasets (default: ./dataset)')
# --- Feature file suffixes (for multi-scale experiments) ---
parser.add_argument('--ms1',  default='',
                    help='Suffix for real-image feature file: "" (default) or "_ms" (multi-scale)')
parser.add_argument('--ms2',  default='',
                    help='Suffix for gen-image feature file: "" (default) or "_ms" (multi-scale)')
# --- SD / Generation ---
parser.add_argument('--sd_2_1', default=True, action="store_true",
                    help='Use Stable Diffusion v2.1 (default: True)')
parser.add_argument('--Ngen',   default=5, type=int,
                    help='Number of generated images per class used in features')
parser.add_argument('--LLM',    default='',
                    help='Prefix for LLM-guided generation: "LLM_" or "" (empty = plain SD)')
# --- Model ---
parser.add_argument('--backbone', default='RN50',
                    help='CLIP backbone: RN50 | RN101 | ViT-B/32 | ViT-B/16 | ViT-L/14')
# --- Misc ---
parser.add_argument('--seed',   default=2024, type=int,
                    help='Random seed for reproducibility')
parser.add_argument('--device', default='cuda:0',
                    help='Device: cpu | cuda:0 | cuda:1 ...')
args = parser.parse_args()

# ===========================================================================
# Reproducibility
# ===========================================================================
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cudnn.benchmark = True

# ===========================================================================
# Load CLIP Model (for metadata only; features are precomputed)
# ===========================================================================
clip_model, preprocess = clip.load(args.backbone, device=args.device)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

# ===========================================================================
# Load Dataset (for class names)
# ===========================================================================
DATASET_MAP = {
    "CUB":      CUBDataset,
    "FLO":      FLODataset,
    "PET":      PETDataset,
    "FOOD":     FOODDataset,
    "ImageNet": ImageNetDataset,
    "EUROSAT":  EUROSATDataset,
}
if args.dataset not in DATASET_MAP:
    raise ValueError(f"Unknown dataset '{args.dataset}'. "
                     f"Choose from: {list(DATASET_MAP.keys())}")
mydataset = DATASET_MAP[args.dataset](args)
all_names = mydataset.all_names

# ===========================================================================
# Resolve Feature File Paths
# ===========================================================================
model_name = args.backbone.replace("-", "").replace("/", "")

# Real-image feature file (produced by vanilla_clip.py or vanilla_clip_ms.py)
CLIP_feature_path = os.path.join(
    args.image_root, args.dataset,
    f"CLIP_{model_name}_feature{args.ms1}.hdf5"
)
# Generated-image feature file (produced by gen_feat.py or muti_scale_gen_feat.py)
CLIP_feature_gen_path = os.path.join(
    args.image_root, args.dataset,
    f"{args.LLM}CLIP_{model_name}_feature_gen{args.Ngen}{args.ms2}.hdf5"
)

# ===========================================================================
# Load Pre-extracted Features
# ===========================================================================
if not os.path.exists(CLIP_feature_path):
    raise FileNotFoundError(
        f"Real-image feature file not found: {CLIP_feature_path}\n"
        f"Run vanilla_clip.py (or vanilla_clip_ms.py with --ms1 _ms) first."
    )
if not os.path.exists(CLIP_feature_gen_path):
    raise FileNotFoundError(
        f"Generated-image feature file not found: {CLIP_feature_gen_path}\n"
        f"Run gen_feat.py (or muti_scale_gen_feat.py with --ms2 _ms) first."
    )

print(" ==> Loading real-image features...")
hf             = h5py.File(CLIP_feature_path, 'r')
test_f         = torch.from_numpy(np.array(hf['test_f'])).float().to(args.device)
test_l         = torch.from_numpy(np.array(hf['test_l'])).float().to(args.device)
all_embeddings = torch.from_numpy(np.array(hf['all_embeddings'])).to(args.device)
print(" ====> Real features loaded.")

print(" ==> Loading generated-image features...")
hf    = h5py.File(CLIP_feature_gen_path, 'r')
gen_f = torch.from_numpy(np.array(hf['gen_f'])).float().to(args.device)
gen_l = torch.from_numpy(np.array(hf['gen_l'])).float().to(args.device)
print(" ====> Gen features loaded.")

# ===========================================================================
# Prototype Computation
# ===========================================================================
# Real prototypes: text embeddings from CLIP text encoder
real_prototypes = all_embeddings  # shape: (C, D)
# Generated prototypes: average of generated image embeddings per class
gen_prototypes  = torch.mean(gen_f, dim=1)   # shape: (C, D)

# Use generated-image prototypes as class classifiers (key contribution of LG-CLIP)
prototypes = gen_prototypes

# ===========================================================================
# Zero-shot Classification
# ===========================================================================
simi_scores       = torch.matmul(test_f, prototypes.T)
predicted_classes = torch.argmax(simi_scores, dim=1)
correct           = torch.sum(predicted_classes == test_l)
Acc               = correct.item() / len(test_l)
print(f"[{args.backbone}] LG-CLIP Acc = {Acc * 100:.2f}%")

# ===========================================================================
# Log Results
# ===========================================================================
log_path = os.path.join(projectPath, "metalog.txt")
log_data = (
    f"Dataset:              {args.dataset}\n"
    f"Backbone:             {args.backbone}\n"
    f"Ngen:                 {args.Ngen}\n"
    f"LLM:                  {args.LLM if args.LLM else '(none)'}\n"
    f"ms1 (real suffix):    {args.ms1 if args.ms1 else '(none)'}\n"
    f"ms2 (gen suffix):     {args.ms2 if args.ms2 else '(none)'}\n"
    f"Real feature path:    {CLIP_feature_path}\n"
    f"Gen  feature path:    {CLIP_feature_gen_path}\n"
    f"Acc:                  {Acc * 100:.2f}%\n"
    f"{'-' * 80}\n"
)
with open(log_path, "a") as lf:
    lf.write(log_data)
print(f"Results appended to {log_path}")
