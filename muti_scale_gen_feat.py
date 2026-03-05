"""
muti_scale_gen_feat.py
-----------------------
Extract multi-scale CLIP visual features from generated images (SD / LLM+SD)
and evaluate zero-shot accuracy at the finest scale.

Workflow:
  1. For each of 10 scale factors (0.1 to 1.0), apply a RandomResizedCrop
     transform to generated images and extract CLIP visual embeddings.
  2. Cache per-scale features in <image_root>/<dataset>/multi_scale_gen/.
  3. Aggregate all scales with triangular weights and L2-normalize to produce
     a final multi-scale feature vector saved as <LLM>CLIP_<backbone>_feature_gen<N>_ms.hdf5.
"""

import os
import json
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


def load_gen_scale_feature(feature_dir, scale_idx, backbone_name, args):
    """
    Load per-scale generated-image features from HDF5.
    File pattern: <LLM>CLIP_<backbone>_feature_gen<N>_scale<idx>.hdf5
    """
    file_path = os.path.join(
        feature_dir,
        f"{args.LLM}CLIP_{backbone_name}_feature_gen{args.Ngen}_scale{scale_idx}.hdf5"
    )
    with h5py.File(file_path, 'r') as f:
        features = torch.from_numpy(np.array(f['gen_f'])).float()
        labels   = torch.from_numpy(np.array(f['gen_l'])).float()
    return features, labels


def save_gen_features(features, labels, save_path):
    """Save aggregated multi-scale generated-image features to HDF5."""
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('gen_f', data=features.cpu().numpy(), compression="gzip")
        f.create_dataset('gen_l', data=labels.cpu().numpy(),   compression="gzip")


def combine_multi_scale_features(base_dir, save_path, backbone_name, args, scales=10):
    """
    Aggregate per-scale features with triangular weights: w_i = i / 55.
    Final features are L2-normalized.
    """
    features_all = None
    labels_all   = None
    for scale in range(1, scales + 1):
        print(f"  Merging scale {scale}/{scales}...")
        features, labels = load_gen_scale_feature(base_dir, scale, backbone_name, args)
        ratio = scale / 55.0
        if features_all is None:
            features_all = features * ratio
            labels_all   = labels
        else:
            features_all += features * ratio
    features_all /= features_all.norm(dim=-1, keepdim=True)
    save_gen_features(features_all, labels_all, save_path)


# ===========================================================================
# Argument Parsing
# ===========================================================================

projectPath = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(
    description="Multi-scale CLIP feature extraction for generated images."
)
# --- Paths ---
parser.add_argument('--dataset',       default='PET',
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
parser.add_argument('--backbone', default='ViT-L/14',
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
# Resolve Paths
# ===========================================================================
model_version  = "XL" if args.sd_xl else "2.1" if args.sd_2_1 else "1.4"
model_name     = args.backbone.replace("-", "").replace("/", "")
exp_identifier = f"{args.LLM}SD_{model_version}_{args.dataset}_10"
# Per-scale cache directory
multi_scale_dir = os.path.join(args.image_root, args.dataset, "multi_scale_gen")
os.makedirs(multi_scale_dir, exist_ok=True)

# Determine source folder for generated images
if args.LLM == "LLM_":
    grp = args.gen_root_path
else:
    grp = os.path.join(projectPath, 'dataset', 'SD_gen')

# ===========================================================================
# Per-scale Feature Extraction
# ===========================================================================
# Compute text embeddings once (for accuracy check at scale 10)
all_embeddings = get_textEmbedding(all_names, clip_model, args)

for scale in range(1, 11):
    scale_factor          = scale / 10.0
    CLIP_feature_gen_path = os.path.join(
        multi_scale_dir,
        f"{args.LLM}CLIP_{model_name}_feature_gen{args.Ngen}_scale{scale}.hdf5"
    )

    if os.path.exists(CLIP_feature_gen_path):
        print(f" ==> Scale {scale}: cache found, skipping extraction.")
        if scale == 10:
            # Load for final accuracy check
            hf    = h5py.File(CLIP_feature_gen_path, 'r')
            gen_f = torch.from_numpy(np.array(hf['gen_f'])).float().to(args.device)
            gen_l = torch.from_numpy(np.array(hf['gen_l'])).float().to(args.device)
        continue

    print(f" ==> Scale {scale}: extracting with scale_factor={scale_factor:.1f} ...")
    # --- Collect image paths ---
    gen_files, gen_labels = [], []
    if args.Ngen == 10:
        # Standard case: enumerate images directly from folder
        for idx, name in enumerate(all_names):
            img_dir = os.path.join(grp, exp_identifier, name)
            for img_file in os.listdir(img_dir):
                gen_files.append(os.path.join(img_dir, img_file))
            gen_labels += [idx] * args.Ngen
    else:
        # Non-standard Ngen: load from JSON index
        json_path = os.path.join(
            args.image_root, args.dataset, "json",
            f"{args.LLM}SD_{model_version}_{args.dataset}_{model_name}_{args.Ngen}.json"
        )
        if not os.path.exists(json_path):
            raise FileNotFoundError(
                f"JSON index file not found: {json_path}\n"
                f"Run text_gen_Ngen_made.py first."
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

    # --- Multi-scale crop transform ---
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

    print(f" ====> Computing visual embeddings for scale {scale}.")
    gen_f, gen_l = get_visualEmbedding(
        clip_model, gendf, args.device, transform=multi_scale_transform
    )
    gen_f = gen_f.reshape(len(all_names), args.Ngen, gen_f.shape[-1])
    gen_l = gen_l.reshape(len(all_names), args.Ngen, 1)
    print(f"  Gen embeddings shape (scale {scale}):", gen_f.shape)

    # --- Save per-scale cache ---
    with h5py.File(CLIP_feature_gen_path, "w") as f:
        f.create_dataset('gen_f',          data=gen_f,               compression="gzip")
        f.create_dataset('gen_l',          data=gen_l,               compression="gzip")
        f.create_dataset('all_embeddings', data=all_embeddings.cpu(), compression="gzip")
    print(f" ====> Scale {scale}: features saved.")

    gen_f = torch.from_numpy(gen_f).float().to(args.device)
    gen_l = torch.from_numpy(gen_l).float().to(args.device)

    # Accuracy check at finest scale (scale=10) as quality sanity check
    if scale == 10:
        gen_f_flat        = gen_f.view(-1, gen_f.size(-1))
        gen_l_flat        = gen_l.view(-1)
        simi_scores       = torch.matmul(gen_f_flat, all_embeddings.T)
        predicted_classes = torch.argmax(simi_scores, dim=1)
        correct           = torch.sum(predicted_classes == gen_l_flat)
        acc               = correct.item() / len(gen_l_flat)
        print(f"[{args.backbone}] Scale-10 (full) Gen Acc = {acc * 100:.2f}%")

# ===========================================================================
# Merge Multi-scale Features
# ===========================================================================
print("\nMerging all scales into a single multi-scale feature file...")
save_ms_path = os.path.join(
    args.image_root, args.dataset,
    f"{args.LLM}CLIP_{model_name}_feature_gen{args.Ngen}_ms.hdf5"
)
combine_multi_scale_features(multi_scale_dir, save_ms_path, model_name, args)
print(f"Multi-scale generated features saved to: {save_ms_path}")
