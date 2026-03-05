"""
muti_scale_merge.py
--------------------
Merge per-scale HDF5 feature files into a single multi-scale feature file
using triangular weighting (w_i = i / sum(1..scales) = i / 55).

This is a standalone utility for cases where the merge step needs to be
re-run without re-extracting features (e.g., after changing the weight scheme).

Supported modes:
  --mode real   : merge real-image multi-scale features (from vanilla_clip_ms.py)
  --mode gen    : merge generated-image multi-scale features (from muti_scale_gen_feat.py)

Usage:
  # Merge real-image features
  python muti_scale_merge.py --mode real --dataset CUB --backbone RN50

  # Merge generated-image features (LLM+SD, Ngen=10)
  python muti_scale_merge.py --mode gen --dataset CUB --backbone RN50 --LLM LLM_ --Ngen 10
"""

import os
import h5py
import re
import torch
import numpy as np
import argparse


# ===========================================================================
# Helper Functions
# ===========================================================================

def load_real_scale(feature_dir: str, scale_idx: int, backbone_name: str):
    """Load real-image features for scale `scale_idx` from HDF5."""
    path = os.path.join(feature_dir, f"CLIP_{backbone_name}_feature_scale{scale_idx}.hdf5")
    with h5py.File(path, 'r') as f:
        features      = torch.from_numpy(np.array(f['test_f'])).float()
        labels        = torch.from_numpy(np.array(f['test_l'])).float()
        all_embedding = np.array(f['all_embeddings'])
    return features, labels, all_embedding


def load_gen_scale(feature_dir: str, scale_idx: int, backbone_name: str, llm: str, ngen: int):
    """Load generated-image features for scale `scale_idx` from HDF5."""
    path = os.path.join(
        feature_dir,
        f"{llm}CLIP_{backbone_name}_feature_gen{ngen}_scale{scale_idx}.hdf5"
    )
    with h5py.File(path, 'r') as f:
        features = torch.from_numpy(np.array(f['gen_f'])).float()
        labels   = torch.from_numpy(np.array(f['gen_l'])).float()
    return features, labels


def save_real_merged(features, labels, all_embedding, save_path: str):
    """Save merged real-image multi-scale features."""
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('test_f',         data=features.cpu().numpy(), compression="gzip")
        f.create_dataset('test_l',         data=labels.cpu().numpy(),   compression="gzip")
        f.create_dataset('all_embeddings', data=all_embedding,          compression="gzip")


def save_gen_merged(features, labels, save_path: str):
    """Save merged generated-image multi-scale features."""
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('gen_f', data=features.cpu().numpy(), compression="gzip")
        f.create_dataset('gen_l', data=labels.cpu().numpy(),   compression="gzip")


# ===========================================================================
# Merge Logic
# ===========================================================================

def merge_real(args, model_name: str, scales: int = 10):
    """Aggregate real-image per-scale features with triangular weights."""
    feature_dir  = os.path.join(args.image_root, args.dataset, "multi_scale")
    save_path    = os.path.join(args.image_root, args.dataset,
                                f"CLIP_{model_name}_feature_ms.hdf5")
    features_all = None
    labels_all   = None
    all_embedding = None

    for scale in range(1, scales + 1):
        print(f"  [real] Loading scale {scale}/{scales}...")
        features, labels, emb = load_real_scale(feature_dir, scale, model_name)
        ratio = scale / 55.0
        if features_all is None:
            features_all  = features * ratio
            labels_all    = labels
            all_embedding = emb
        else:
            features_all += features * ratio

    features_all /= features_all.norm(dim=-1, keepdim=True)
    save_real_merged(features_all, labels_all, all_embedding, save_path)
    print(f"Real multi-scale features saved: {save_path}")


def merge_gen(args, model_name: str, scales: int = 10):
    """Aggregate generated-image per-scale features with triangular weights."""
    feature_dir  = os.path.join(args.image_root, args.dataset, "multi_scale_gen")
    save_path    = os.path.join(args.image_root, args.dataset,
                                f"{args.LLM}CLIP_{model_name}_feature_gen{args.Ngen}_ms.hdf5")
    features_all = None
    labels_all   = None

    for scale in range(1, scales + 1):
        print(f"  [gen] Loading scale {scale}/{scales}...")
        features, labels = load_gen_scale(feature_dir, scale, model_name, args.LLM, args.Ngen)
        ratio = scale / 55.0
        if features_all is None:
            features_all = features * ratio
            labels_all   = labels
        else:
            features_all += features * ratio

    features_all /= features_all.norm(dim=-1, keepdim=True)
    save_gen_merged(features_all, labels_all, save_path)
    print(f"Gen multi-scale features saved: {save_path}")


# ===========================================================================
# Argument Parsing
# ===========================================================================

projectPath = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(
    description="Merge per-scale CLIP features into a single multi-scale feature file."
)
parser.add_argument('--mode',       default='real', choices=['real', 'gen'],
                    help='Feature type to merge: "real" or "gen" (default: real)')
parser.add_argument('--dataset',    default='CUB',
                    help='Dataset name: CUB | FLO | PET | FOOD | ImageNet | EUROSAT')
parser.add_argument('--image_root', default=os.path.join(projectPath, 'dataset'),
                    help='Root directory of datasets (default: ./dataset)')
parser.add_argument('--backbone',   default='RN50',
                    help='CLIP backbone name (must match the per-scale files)')
parser.add_argument('--LLM',        default='',
                    help='LLM prefix for gen mode: "LLM_" or "" (default: "")')
parser.add_argument('--Ngen',       default=10, type=int,
                    help='Ngen for gen mode (default: 10)')
args = parser.parse_args()

model_name = args.backbone.replace("-", "").replace("/", "")

if args.mode == 'real':
    merge_real(args, model_name)
else:
    merge_gen(args, model_name)

print("Done.")
