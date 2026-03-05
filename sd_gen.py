"""
sd_gen.py
---------
Generate images using Stable Diffusion for each class in a dataset.
Prompts are simple templates: "A photo of a <classname>."

Supports:
  - SD v1.4  (CompVis/stable-diffusion-v1-4)
  - SD v2.1  (stabilityai/stable-diffusion-2-1-base)  [default]
  - SD XL    (stabilityai/stable-diffusion-xl-base-1.0)

Generated images are stored under:
  <gen_root_path>/<exp_identifier>/<class_name>/<seed>_<class_name>.jpg

Resumable: already-generated images are detected and skipped.

Usage:
  python sd_gen.py --dataset CUB --Ngen 10 [--sd_2_1 | --sd_xl] [--device cuda:0]
"""

import os
import re
import torch
import random
import logging
import warnings
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
import torch.backends.cudnn as cudnn
from diffusers import StableDiffusionPipeline

from utils.myDataset import (
    CUBDataset, FLODataset, PETDataset, FOODDataset,
    ImageNetDataset, EUROSATDataset
)
from utils.helper_func import numpy_to_pil

warnings.filterwarnings('ignore')


def dummy_safety_checker(images, clip_input=None):
    """Bypass the NSFW safety checker so all images are returned as-is."""
    return images, False


# ===========================================================================
# Argument Parsing
# ===========================================================================

projectPath = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(
    description="Generate images with Stable Diffusion using class-name prompts."
)
# --- Dataset ---
parser.add_argument('--dataset',       default='ImageNet',
                    help='Dataset name: CUB | FLO | PET | FOOD | ImageNet | EUROSAT')
parser.add_argument('--image_root',    default=os.path.join(projectPath, 'dataset'),
                    help='Root directory containing all datasets (default: ./dataset)')
parser.add_argument('--gen_root_path', default=os.path.join(projectPath, 'dataset', 'SD_gen'),
                    help='Output directory for generated images (default: ./dataset/SD_gen)')
# --- SD Version (mutually exclusive flags) ---
parser.add_argument('--sd_2_1', default=True,  action="store_true",
                    help='Use SD v2.1 (stabilityai/stable-diffusion-2-1-base) [default]')
parser.add_argument('--sd_xl',  default=False, action="store_true",
                    help='Use SD XL (stabilityai/stable-diffusion-xl-base-1.0)')
# --- Generation ---
parser.add_argument('--Ngen',   default=10, type=int,
                    help='Number of images to generate per class (default: 10)')
# --- Misc ---
parser.add_argument('--seed',   default=2024, type=int,
                    help='Base random seed for reproducibility')
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
# Experiment Setup & Logging
# ===========================================================================
model_version  = "XL" if args.sd_xl else "2.1" if args.sd_2_1 else "1.4"
exp_identifier = f"SD_{model_version}_{args.dataset}_{args.Ngen}"
print(f"==> Starting generation: {exp_identifier}")

# Create output + log directory
eval_log_path = os.path.join(args.gen_root_path, exp_identifier)
Path(eval_log_path).mkdir(parents=True, exist_ok=True)
log_file = os.path.join(eval_log_path, "sd_gen_log.txt")

logging.basicConfig(
    format='%(message)s', level=logging.INFO,
    filename=log_file, filemode='w'
)
logger = logging.getLogger(__name__)
for k, v in args.__dict__.items():
    logger.info(f"{k}: {v}")
logger.info("=" * 50)

# ===========================================================================
# Select SD Model
# ===========================================================================
if args.sd_xl:
    pretrained_model = "stabilityai/stable-diffusion-xl-base-1.0"
elif args.sd_2_1:
    pretrained_model = "stabilityai/stable-diffusion-2-1-base"
else:
    pretrained_model = "CompVis/stable-diffusion-v1-4"

# ===========================================================================
# Image Generation Loop
# ===========================================================================
for class_label, class_name in enumerate(all_names):
    img_dir = os.path.join(args.gen_root_path, exp_identifier, class_name)
    Path(img_dir).mkdir(parents=True, exist_ok=True)

    # Check how many images already exist (resume support)
    existing_images = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
    num_existing    = len(existing_images)
    if num_existing >= args.Ngen:
        msg = f"[{class_label}/{len(all_names)}] '{class_name}': already done ({num_existing}/{args.Ngen}). Skipping."
        print(msg); logger.info(msg)
        continue

    msg = f"====> [{class_label}/{len(all_names)}] Generating '{class_name}'"
    print(msg); logger.info(msg)

    # Load pipeline (loaded fresh per class to avoid memory accumulation)
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model, safety_checker=None
    ).to(args.device)

    # Simple class-name prompt (plain SD, no LLM enrichment)
    prompt = f"A photo of a {class_name}."
    logger.info(f"  Prompt: '{prompt}'")

    # Determine starting index for remaining images
    existing_indices = [
        int(re.search(r'^(\d+)_', img).group(1))
        for img in existing_images
        if re.match(r'^\d+_', img)
    ]
    start_idx = max(existing_indices) + 1 if existing_indices else 0

    for i in range(start_idx, args.Ngen):
        generator = torch.Generator(device=args.device)
        generator.manual_seed(i)
        image_out = pipeline(prompt, output_type="pt", generator=generator)[0]
        img_path  = os.path.join(img_dir, f"{i}_{class_name}.jpg")
        numpy_to_pil(
            image_out.permute(0, 2, 3, 1).cpu().detach().numpy()
        )[0].save(img_path, "JPEG")
        log_msg = f"  Saved: {img_path}"
        print(log_msg); logger.info(log_msg)
