"""
gen_feat.py
-----------
Extract CLIP visual and text features from generated images (SD / LLM+SD)
and evaluate the generation quality via zero-shot classification accuracy.

Workflow:
  1. Load or extract text embeddings for all class names.
  2. Enumerate generated images in <gen_root_path>/<exp_identifier>/<class_name>/
     OR load paths from a JSON file (when Ngen != 10).
  3. Extract L2-normalized visual embeddings via CLIP.
  4. Compute cosine-similarity-based zero-shot accuracy on generated images.
  5. Cache features in HDF5 for reuse by mega.py.
"""

import os
import json
import h5py
import torch
import random
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from PIL import Image

import clip
from utils.myDataset import (
    CUBDataset, FLODataset, PETDataset, FOODDataset,
    ImageNetDataset, EUROSATDataset
)


# ===========================================================================
# Feature Extraction Helpers
# ===========================================================================

def get_textEmbedding(classnames, clip_model, args, norm=True):
    """
    Compute L2-normalized CLIP text embeddings for each class name.
    Template: "A photo of a <classname>."
    """
    with torch.no_grad():
        classnames = [c.replace('_', ' ') for c in classnames]
        texts  = [f"A photo of a {c}." for c in classnames]
        tokens = clip.tokenize(texts, context_length=77).to(args.device)
        feats  = clip_model.encode_text(tokens).float().to(args.device)
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
            img  = img.unsqueeze(0).to(device)
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
    description="Extract CLIP features from generated images and evaluate quality."
)
# --- Paths ---
parser.add_argument('--dataset',       default='ImageNet',
                    help='Dataset name: CUB | FLO | PET | FOOD | ImageNet | EUROSAT')
parser.add_argument('--image_root',    default=os.path.join(projectPath, 'dataset'),
                    help='Root directory containing all datasets (default: ./dataset)')
parser.add_argument('--gen_root_path', default=os.path.join(projectPath, 'dataset', 'LLM_SD_gen'),
                    help='Root directory for LLM-guided generated images (default: ./dataset/LLM_SD_gen)')
# --- SD Version ---
parser.add_argument('--sd_2_1',  default=True, action="store_true",
                    help='Use Stable Diffusion v2.1 (default: True)')
parser.add_argument('--sd_xl',   default=False, action="store_true",
                    help='Use Stable Diffusion XL (overrides --sd_2_1)')
# --- Generation ---
parser.add_argument('--Ngen',    default=10, type=int,
                    help='Number of generated images per class (default: 10)')
parser.add_argument('--LLM',     default='',
                    help='Prefix for LLM-guided generation: "LLM_" or "" (empty = plain SD)')
# --- Model ---
parser.add_argument('--backbone', default='RN50',
                    help='CLIP backbone: RN50 | RN101 | ViT-B/32 | ViT-B/16 | ViT-L/14')
# --- Misc ---
parser.add_argument('--seed',    default=2024, type=int,
                    help='Random seed for reproducibility')
parser.add_argument('--device',  default='cuda:0',
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
# Resolve Paths & Cache File
# ===========================================================================
model_version  = "XL" if args.sd_xl else "2.1" if args.sd_2_1 else "1.4"
# e.g. "LLM_SD_2.1_CUB_10" or "SD_2.1_CUB_10"
exp_identifier = f"{args.LLM}SD_{model_version}_{args.dataset}_{args.Ngen}"
model_name     = args.backbone.replace("-", "").replace("/", "")
# HDF5 cache file for generated image features
CLIP_feature_gen_path = os.path.join(
    args.image_root, args.dataset,
    f"{args.LLM}CLIP_{model_name}_feature_gen{args.Ngen}.hdf5"
)

# ===========================================================================
# Feature Extraction / Loading
# ===========================================================================
if os.path.exists(CLIP_feature_gen_path):
    print(" ==> Loading cached generated-image features...")
    hf             = h5py.File(CLIP_feature_gen_path, 'r')
    gen_f          = torch.from_numpy(np.array(hf['gen_f'])).float().to(args.device)
    gen_l          = torch.from_numpy(np.array(hf['gen_l'])).float().to(args.device)
    all_embeddings = torch.from_numpy(np.array(hf['all_embeddings'])).to(args.device)
    print(" ====> Cache loaded.")
else:
    print(" ==> Extracting features from generated images...")
    # --- Text embeddings ---
    print(" ====> Computing text embeddings from CLIP text encoder.")
    all_embeddings = get_textEmbedding(all_names, clip_model, args)
    print("  Text embeddings shape:", all_embeddings.shape)

    # --- Determine source folder for generated images ---
    if args.LLM == "LLM_":
        # LLM-guided generation: images in gen_root_path
        grp = args.gen_root_path
    else:
        # Plain SD generation: images in dataset/SD_gen
        grp = os.path.join(projectPath, 'dataset', 'SD_gen')

    # --- Collect generated image paths ---
    gen_files, gen_labels = [], []
    if args.Ngen == 10:
        # Standard case: enumerate images directly from folder
        for idx, name in enumerate(all_names):
            img_dir = os.path.join(grp, exp_identifier, name)
            for img_file in os.listdir(img_dir):
                gen_files.append(os.path.join(img_dir, img_file))
            gen_labels += [idx] * args.Ngen
    else:
        # Non-standard Ngen: read from a pre-generated JSON index file
        json_path = os.path.join(
            args.image_root, args.dataset, "json",
            f"{args.LLM}SD_{model_version}_{args.dataset}_{model_name}_{args.Ngen}.json"
        )
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"JSON index file not found: {json_path}\n"
                f"Run text_gen_Ngen_made.py first to generate it."
            )
        with open(json_path, 'r', encoding='utf-8') as jf:
            json_data = json.load(jf)
        for idx, name in enumerate(all_names):
            for entry in json_data[name]:
                gen_files.append(entry['image'])
                gen_labels.append(idx)

    assert len(gen_files) == len(all_names) * args.Ngen, \
        f"Expected {len(all_names) * args.Ngen} images, got {len(gen_files)}"

    gen_labels = np.array(gen_labels)
    gendf      = list(zip(gen_files, gen_labels))

    # --- Visual embeddings ---
    print(" ====> Computing visual embeddings from CLIP image encoder.")
    gen_f, gen_l = get_visualEmbedding(clip_model, gendf, args.device, transform=preprocess)
    gen_f = gen_f.reshape(len(all_names), args.Ngen, gen_f.shape[-1])
    gen_l = gen_l.reshape(len(all_names), args.Ngen, 1)
    print("  Gen embeddings shape:", gen_f.shape)

    # --- Save to HDF5 cache ---
    os.makedirs(os.path.dirname(CLIP_feature_gen_path), exist_ok=True)
    with h5py.File(CLIP_feature_gen_path, "w") as f:
        f.create_dataset('gen_f',          data=gen_f,               compression="gzip")
        f.create_dataset('gen_l',          data=gen_l,               compression="gzip")
        f.create_dataset('all_embeddings', data=all_embeddings.cpu(), compression="gzip")
    print(" ====> Features saved to cache.")

    gen_f = torch.from_numpy(gen_f).float().to(args.device)
    gen_l = torch.from_numpy(gen_l).float().to(args.device)

# ===========================================================================
# Zero-shot Classification on Generated Images
# ===========================================================================
# Flatten (C, Ngen, D) -> (C*Ngen, D) for classification
gen_f_flat        = gen_f.view(-1, gen_f.size(-1))
gen_l_flat        = gen_l.view(-1)
simi_scores       = torch.matmul(gen_f_flat, all_embeddings.T)
predicted_classes = torch.argmax(simi_scores, dim=1)
correct           = torch.sum(predicted_classes == gen_l_flat)
acc               = correct.item() / len(gen_l_flat)
print(f"[{args.backbone}] Gen-image Zero-shot Acc = {acc * 100:.2f}%")
