"""
vanilla_clip.py
---------------
Extract CLIP visual and text features from real images and evaluate zero-shot
classification accuracy using cosine similarity between image and text embeddings.

Features are saved to an HDF5 file for reuse. If the cache file already exists,
it is loaded directly to avoid redundant computation.
"""

import os
import h5py
import torch
import random
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from PIL import Image
from tqdm import tqdm

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
    Compute CLIP text embeddings for each class name.
    Each class is formatted as "A photo of a <classname>."
    Returns L2-normalized embeddings by default.
    """
    with torch.no_grad():
        classnames = [c.replace('_', ' ') for c in classnames]
        text_descriptions = [f"A photo of a {c}." for c in classnames]
        text_tokens = clip.tokenize(text_descriptions, context_length=77).to(args.device)
        text_features = clip_model.encode_text(text_tokens).float().to(args.device)
        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features


def get_visualEmbedding(clip_model, dataframe, device, transform=None):
    """
    Compute L2-normalized CLIP visual embeddings for a list of (image_path, label) pairs.
    Returns:
        features: np.ndarray of shape (N, D)
        labels:   np.ndarray of shape (N,)
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


# ===========================================================================
# Argument Parsing
# ===========================================================================

projectPath = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(
    description="Evaluate zero-shot CLIP accuracy on real images."
)
# --- Paths ---
parser.add_argument('--dataset',    default='FLO',
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
                    help='Device to use: cpu | cuda:0 | cuda:1 ...')
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
# Feature Extraction / Loading
# ===========================================================================
model_name = args.backbone.replace("-", "").replace("/", "")
# Cache file stores train/test features and text embeddings
CLIP_feature_path = os.path.join(
    args.image_root, args.dataset, f"CLIP_{model_name}_feature.hdf5"
)

if os.path.exists(CLIP_feature_path):
    print(" ==> Load existing feature cache.")
    hf = h5py.File(CLIP_feature_path, 'r')
    test_f        = torch.from_numpy(np.array(hf['test_f'])).float().to(args.device)
    test_l        = torch.from_numpy(np.array(hf['test_l'])).float().to(args.device)
    all_embeddings = torch.from_numpy(np.array(hf['all_embeddings'])).to(args.device)
    print(" ====> Feature cache loaded.")
else:
    print(" ==> Extracting features from scratch...")
    # Text embeddings
    print(" ====> Computing text embeddings from CLIP text encoder.")
    all_embeddings = get_textEmbedding(all_names, clip_model, args)
    print("  Text embeddings shape:", all_embeddings.shape)

    # Visual embeddings (train + test)
    traindf = list(zip(mydataset.train_files, mydataset.train_labels.numpy()))
    testdf  = list(zip(mydataset.test_files,  mydataset.test_labels.numpy()))
    print(" ====> Computing visual embeddings from CLIP image encoder.")
    train_f, train_l = get_visualEmbedding(clip_model, traindf, args.device, preprocess)
    test_f,  test_l  = get_visualEmbedding(clip_model, testdf,  args.device, preprocess)
    print("  Train embeddings shape:", train_f.shape)
    print("  Test  embeddings shape:", test_f.shape)

    # Save to HDF5 cache
    os.makedirs(os.path.dirname(CLIP_feature_path), exist_ok=True)
    with h5py.File(CLIP_feature_path, "w") as f:
        f.create_dataset('train_f',        data=train_f,              compression="gzip")
        f.create_dataset('train_l',        data=train_l,              compression="gzip")
        f.create_dataset('test_f',         data=test_f,               compression="gzip")
        f.create_dataset('test_l',         data=test_l,               compression="gzip")
        f.create_dataset('all_embeddings', data=all_embeddings.cpu(), compression="gzip")
    print(" ====> Feature cache saved.")

    test_f         = torch.from_numpy(test_f).float().to(args.device)
    test_l         = torch.from_numpy(test_l).float().to(args.device)

# ===========================================================================
# Zero-shot Classification
# ===========================================================================
simi_scores      = torch.matmul(test_f, all_embeddings.T)
predicted_classes = torch.argmax(simi_scores, dim=1)
correct           = torch.sum(predicted_classes == test_l)
acc               = correct.item() / len(test_l)
print(f"[{args.backbone}] Zero-shot Acc = {acc * 100:.2f}%")
