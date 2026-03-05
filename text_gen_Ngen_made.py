"""
text_gen_Ngen_made.py
---------------------
Dynamic filtering of generated images based on CLIP classification confidence.

Given a set of N_all=10 generated images per class, this script:
  1. Computes CLIP text embeddings for all class names.
  2. Computes CLIP visual embeddings for all generated images.
  3. Ranks images by cosine similarity to their ground-truth class text embedding.
  4. Selects the top-Ngen images per class, prioritizing correctly classified ones.
  5. Saves the filtered image paths and similarity scores to a JSON index file.

The output JSON is consumed by gen_feat.py and muti_scale_gen_feat.py
when --Ngen != 10 (e.g., Ngen=5).

Usage:
  python text_gen_Ngen_made.py \\
      --dataset CUB --Ngen 5 --LLM LLM_ [--backbone ViT-B/32] [--device cuda:0]
"""

import os
import json
import torch
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


# ===========================================================================
# Argument Parsing
# ===========================================================================

projectPath = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(
    description="Select top-Ngen generated images per class based on CLIP similarity."
)
# --- Dataset ---
parser.add_argument('--dataset',       default='ImageNet',
                    help='Dataset name: CUB | FLO | PET | FOOD | ImageNet | EUROSAT')
parser.add_argument('--image_root',    default=os.path.join(projectPath, 'dataset'),
                    help='Root directory of datasets (default: ./dataset)')
parser.add_argument('--gen_root_path', default=os.path.join(projectPath, 'dataset', 'SD_gen'),
                    help='Root directory of generated images (default: ./dataset/SD_gen)')
# --- SD Version ---
parser.add_argument('--sd_2_1', default=True, action="store_true",
                    help='Generated images from SD v2.1 (default)')
# --- Selection ---
parser.add_argument('--Ngen',   default=10, type=int,
                    help='Number of images to select per class (default: 10)')
parser.add_argument('--LLM',    default='LLM_',
                    help='Prefix for LLM-guided generation: "LLM_" or "" (plain SD)')
# --- Model ---
parser.add_argument('--backbone', default='ViT-B/32',
                    help='CLIP backbone for scoring: RN50 | ViT-B/32 | ViT-L/14')
# --- Misc ---
parser.add_argument('--device', default='cuda:0',
                    help='Device: cpu | cuda:0 | cuda:1 ...')
args = parser.parse_args()

# ===========================================================================
# Reproducibility
# ===========================================================================
torch.manual_seed(2024)
np.random.seed(2024)
cudnn.benchmark = True

# ===========================================================================
# Load CLIP
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
# Compute Text Embeddings
# ===========================================================================
text_embeddings = get_textEmbedding(all_names, clip_model, args)

# ===========================================================================
# Collect Generated Image Paths (N_all = 10 per class)
# ===========================================================================
model_version = "2.1" if args.sd_2_1 else "1.4"

# Determine source folder based on LLM flag
if args.LLM == "LLM_":
    grp = args.gen_root_path  # LLM_SD_gen
else:
    grp = os.path.join(projectPath, 'dataset', 'SD_gen')

# The full generation folder contains all 10 images per class
gen_root_dir = os.path.join(grp, f"{args.LLM}SD_{model_version}_{args.dataset}_10")
if not os.path.exists(gen_root_dir):
    raise FileNotFoundError(
        f"Generated image folder not found: {gen_root_dir}\n"
        f"Run sd_gen.py or llm_sd_gen.py first with Ngen=10."
    )

gen_files, gen_labels = [], []
for idx, name in enumerate(all_names):
    img_dir = os.path.join(gen_root_dir, name)
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Class folder missing: {img_dir}")
    for img_file in os.listdir(img_dir):
        gen_files.append(os.path.join(img_dir, img_file))
    gen_labels += [idx] * 10   # N_all = 10

assert len(gen_files) == len(all_names) * 10, \
    f"Expected {len(all_names) * 10} images, found {len(gen_files)}"

# ===========================================================================
# Compute Visual Embeddings
# ===========================================================================
gen_labels_np = np.array(gen_labels)
gendf         = list(zip(gen_files, gen_labels_np))
gen_f, gen_l  = get_visualEmbedding(clip_model, gendf, args.device, transform=preprocess)
gen_f  = torch.from_numpy(gen_f).float().to(args.device)
gen_l  = torch.from_numpy(gen_l).long().to(args.device)

# ===========================================================================
# Classify and Score
# ===========================================================================
simi_scores       = torch.matmul(gen_f, text_embeddings.T)   # (N, C)
predicted_classes = torch.argmax(simi_scores, dim=1)          # (N,)

# ===========================================================================
# Per-class Filtering: prefer correct predictions, sort by confidence
# ===========================================================================
result_dict = {i: [] for i in range(len(all_names))}
for img_idx, (gen_file, label) in enumerate(zip(gen_files, gen_l)):
    label_int = int(label.item())
    prob      = simi_scores[img_idx, label_int].item()
    result_dict[label_int].append({"image": gen_file, "probability": prob})

filtered_result_dict = {}
for label_int, images in result_dict.items():
    # Split into correctly and incorrectly classified images
    correct_imgs   = [img for img in images
                      if predicted_classes[gen_files.index(img["image"])] == label_int]
    incorrect_imgs = [img for img in images if img not in correct_imgs]

    # Sort each group by CLIP similarity (descending)
    correct_imgs   = sorted(correct_imgs,   key=lambda x: x["probability"], reverse=True)
    incorrect_imgs = sorted(incorrect_imgs, key=lambda x: x["probability"], reverse=True)

    # Select top-Ngen, prioritizing correct predictions
    selected = correct_imgs[:args.Ngen]
    if len(selected) < args.Ngen:
        selected += incorrect_imgs[:args.Ngen - len(selected)]

    filtered_result_dict[label_int] = selected

# ===========================================================================
# Save JSON Index
# ===========================================================================
backbone_name   = args.backbone.replace("/", "").replace("-", "")
json_output_dir = os.path.join(args.image_root, args.dataset, "json")
os.makedirs(json_output_dir, exist_ok=True)
json_output_path = os.path.join(
    json_output_dir,
    f"{args.LLM}SD_{model_version}_{args.dataset}_{backbone_name}_{args.Ngen}.json"
)
json_result = {
    all_names[label]: filtered_result_dict[label]
    for label in filtered_result_dict
}
with open(json_output_path, 'w', encoding='utf-8') as jf:
    json.dump(json_result, jf, indent=4, ensure_ascii=False)

print(f"Filtered JSON index saved to: {json_output_path}")
