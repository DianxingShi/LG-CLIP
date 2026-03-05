"""
vanilla_clip_ms.py
------------------
Extract multi-scale CLIP visual features from real (test) images and evaluate
zero-shot classification accuracy at each scale.

Strategy:
  - Apply 10 different RandomResizedCrop scales (0.1 to 1.0) to test images.
  - Save per-scale HDF5 feature files under <image_root>/<dataset>/multi_scale/.
  - Aggregate all scales with a triangular weighting (scale_i / 55) and
    L2-normalize to produce a final multi-scale feature vector.
  - Save the combined multi-scale features for downstream use in mega.py.
"""

import os
import re
import h5py
import torch
import random
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms

import clip
from utils.myDataset import (
    CUBDataset, FLODataset, PETDataset, FOODDataset,
    ImageNetDataset, EUROSATDataset
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ===========================================================================
# Feature Extraction Helpers
# ===========================================================================

def get_textEmbedding(classnames, clip_model, args, norm=True):
    """
    Compute L2-normalized CLIP text embeddings for a list of class names.
    Template: "A photo of a <classname>."
    """
    with torch.no_grad():
        classnames = [c.replace('_', ' ') for c in classnames]
        texts = [f"A photo of a {c}." for c in classnames]
        tokens = clip.tokenize(texts, context_length=77).to(args.device)
        feats = clip_model.encode_text(tokens).float().to(args.device)
        if norm:
            feats /= feats.norm(dim=-1, keepdim=True)
    return feats


def get_visualEmbedding(clip_model, dataframe, device, transform=None):
    """
    Compute L2-normalized CLIP visual embeddings for (image_path, label) pairs.
    Returns:
        features: np.ndarray (N, D)
        labels:   np.ndarray (N,)
    """
    with torch.no_grad():
        features, labels = [], []
        progress = tqdm(total=len(dataframe), ncols=100)
        for img_path, label in dataframe:
            progress.update(1)
            img = Image.open(img_path).convert('RGB')
            if transform is not None:
                img = transform(img)
            img = img.unsqueeze(0).to(device)
            feat = clip_model.encode_image(img).float().to(device)
            feat /= feat.norm(dim=-1, keepdim=True)
            features.append(feat.cpu())
            labels.append(label)
        progress.close()
    return np.concatenate(features, axis=0), np.array(labels)


def load_scale_feature(feature_dir, scale_idx, backbone_name):
    """
    Load per-scale features from HDF5 for real (test) images.
    File pattern: CLIP_<backbone>_feature_scale<idx>.hdf5
    """
    file_path = os.path.join(
        feature_dir, f"CLIP_{backbone_name}_feature_scale{scale_idx}.hdf5"
    )
    with h5py.File(file_path, 'r') as f:
        features      = torch.from_numpy(np.array(f['test_f'])).float()
        labels        = torch.from_numpy(np.array(f['test_l'])).float()
        all_embedding = np.array(f['all_embeddings'])
    return features, labels, all_embedding


def save_scale_feature(features, labels, all_embedding, save_path):
    """Save multi-scale aggregated features and labels to HDF5."""
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('test_f',         data=features.cpu().numpy(),  compression="gzip")
        f.create_dataset('test_l',         data=labels.cpu().numpy(),    compression="gzip")
        f.create_dataset('all_embeddings', data=all_embedding,           compression="gzip")


def combine_multi_scale_features(base_dir, save_path, backbone_name, scales=10):
    """
    Aggregate per-scale features using triangular weights: w_i = i / sum(1..scales).
    sum(1..10) = 55, so w_i = i / 55.
    Final features are L2-normalized after aggregation.
    """
    features_all = None
    labels_all   = None

    for scale in range(1, scales + 1):
        print(f"  Merging scale {scale}/{scales}...")
        features, labels, all_embedding = load_scale_feature(base_dir, scale, backbone_name)
        ratio = scale / 55.0  # triangular weight
        if features_all is None:
            features_all = features * ratio
            labels_all   = labels
        else:
            features_all += features * ratio

    # L2 normalize the aggregated features
    features_all /= features_all.norm(dim=-1, keepdim=True)
    save_scale_feature(features_all, labels_all, all_embedding, save_path)


# ===========================================================================
# Argument Parsing
# ===========================================================================

projectPath = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(
    description="Multi-scale CLIP feature extraction for real images."
)
# --- Paths ---
parser.add_argument('--dataset',    default='ImageNet',
                    help='Dataset name: CUB | FLO | PET | FOOD | ImageNet | EUROSAT')
parser.add_argument('--image_root', default=os.path.join(projectPath, 'dataset'),
                    help='Root directory containing all datasets (default: ./dataset)')
# --- Model ---
parser.add_argument('--backbone',   default='RN50',
                    help='CLIP backbone: RN50 | RN101 | ViT-B/32 | ViT-B/16 | ViT-L/14')
# --- Misc ---
parser.add_argument('--seed',       default=2024, type=int,
                    help='Random seed for reproducibility')
parser.add_argument('--device',     default='cuda:0',
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
# Load CLIP Model
# ===========================================================================
clip_model, preprocess = clip.load(args.backbone, device=args.device)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

# ===========================================================================
# Load Dataset
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
# Multi-scale Feature Extraction
# ===========================================================================
model_name       = args.backbone.replace("-", "").replace("/", "")
multi_scale_dir  = os.path.join(args.image_root, args.dataset, "multi_scale")
os.makedirs(multi_scale_dir, exist_ok=True)

# Compute text embeddings once (shared across all scales)
all_embeddings = get_textEmbedding(all_names, clip_model, args)
print("Text embeddings shape:", all_embeddings.shape)

testdf = list(zip(mydataset.test_files, mydataset.test_labels.numpy()))

for scale in range(1, 11):
    scale_factor      = scale / 10.0
    CLIP_feature_path = os.path.join(
        multi_scale_dir, f"CLIP_{model_name}_feature_scale{scale}.hdf5"
    )
    if os.path.exists(CLIP_feature_path):
        print(f" ==> Scale {scale}: cache found, skipping extraction.")
        # Still need features for accuracy report
        with h5py.File(CLIP_feature_path, 'r') as hf:
            test_f = torch.from_numpy(np.array(hf['test_f'])).float().to(args.device)
            test_l = torch.from_numpy(np.array(hf['test_l'])).float().to(args.device)
    else:
        print(f" ==> Scale {scale}: extracting with scale_factor={scale_factor:.1f} ...")
        # Multi-scale crop transform: force crop to the given area ratio
        multi_scale_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                size=224,
                scale=(scale_factor, scale_factor),
                interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])
        test_f, test_l = get_visualEmbedding(
            clip_model, testdf, args.device, transform=multi_scale_transform
        )
        print(f"  Test embeddings shape (scale {scale}):", test_f.shape)

        # Save per-scale cache
        with h5py.File(CLIP_feature_path, "w") as f:
            f.create_dataset('test_f',         data=test_f,               compression="gzip")
            f.create_dataset('test_l',         data=test_l,               compression="gzip")
            f.create_dataset('all_embeddings', data=all_embeddings.cpu(), compression="gzip")
        print(f" ====> Scale {scale}: features saved.")

        test_f = torch.from_numpy(test_f).float().to(args.device)
        test_l = torch.from_numpy(test_l).float().to(args.device)

    # Per-scale classification accuracy (sanity check)
    simi_scores       = torch.matmul(test_f, all_embeddings.T)
    predicted_classes = torch.argmax(simi_scores, dim=1)
    correct           = torch.sum(predicted_classes == test_l)
    acc               = correct.item() / len(test_l)
    print(f"  [{args.backbone}] Scale-{scale} Acc = {acc * 100:.2f}%")

# ===========================================================================
# Merge Multi-scale Features (triangular weighting)
# ===========================================================================
print("\nMerging all scales into a single multi-scale feature file...")
save_ms_path = os.path.join(
    args.image_root, args.dataset, f"CLIP_{model_name}_feature_ms.hdf5"
)
combine_multi_scale_features(multi_scale_dir, save_ms_path, model_name)
print(f"Multi-scale features saved to: {save_ms_path}")
