"""
sd_xl_gen.py
------------
Generate images using Stable Diffusion XL (or SD 2.1 / 1.4) with accelerate
for multi-GPU / mixed-precision support.

Prompts are simple class-name templates: "A photo of a <classname>."
Images are stored under:
  <gen_root_path>/<exp_identifier>/<class_name>/<seed>_<class_name>.jpg

Usage:
  python sd_xl_gen.py --dataset FLO --Ngen 10 --sd_xl [--device cuda:0]
"""

import os
import torch
import random
import logging
import warnings
import argparse
import torchvision.transforms as T
from pathlib import Path
from diffusers import AutoPipelineForText2Image
from accelerate import Accelerator

from utils.myDataset import (
    CUBDataset, FLODataset, PETDataset, FOODDataset,
    ImageNetDataset, EUROSATDataset
)

warnings.filterwarnings('ignore')

# Allow xformers memory-efficient attention if available
os.environ["XFORMERS_OFFLOAD"]    = "1"
os.environ["XFORMERS_THRESHOLD"]  = "1"


def dummy_safety_checker(images, clip_input=None):
    """Bypass the NSFW safety checker so all images are returned."""
    return images, False


# ===========================================================================
# Argument Parsing
# ===========================================================================

projectPath = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(
    description="Generate images with SD XL (or SD 2.1/1.4) using accelerate."
)
# --- Dataset ---
parser.add_argument('--dataset',       default='FLO',
                    help='Dataset name: CUB | FLO | PET | FOOD | ImageNet | EUROSAT')
parser.add_argument('--image_root',    default=os.path.join(projectPath, 'dataset'),
                    help='Root directory of datasets (default: ./dataset)')
parser.add_argument('--gen_root_path', default=os.path.join(projectPath, 'dataset', 'SD_gen'),
                    help='Output directory for generated images (default: ./dataset/SD_gen)')
# --- SD Version ---
parser.add_argument('--sd_2_1', default=False, action="store_true",
                    help='Use SD v2.1 instead of SD XL')
parser.add_argument('--sd_xl',  default=True,  action="store_true",
                    help='Use SD XL (default)')
# --- Generation ---
parser.add_argument('--Ngen',   default=10, type=int,
                    help='Number of images to generate per class (default: 10)')
# --- Misc ---
parser.add_argument('--seed',   default=2024, type=int,
                    help='Base random seed')
parser.add_argument('--device', default='cuda:0',
                    help='Device hint (actual device determined by accelerate)')
args = parser.parse_args()

# ===========================================================================
# Set Random Seed
# ===========================================================================
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

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

eval_log_path = os.path.join(args.gen_root_path, exp_identifier)
Path(eval_log_path).mkdir(parents=True, exist_ok=True)
log_file = os.path.join(eval_log_path, "sd_gen_log.txt")
logging.basicConfig(filename=log_file, filemode='w', level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================================================================
# Select SD Model & Load Pipeline via Accelerate
# ===========================================================================
if args.sd_xl:
    pretrained_model = "stabilityai/stable-diffusion-xl-base-1.0"
elif args.sd_2_1:
    pretrained_model = "stabilityai/stable-diffusion-2-1-base"
else:
    pretrained_model = "CompVis/stable-diffusion-v1-4"

accelerator = Accelerator()
device      = accelerator.device
pipeline    = AutoPipelineForText2Image.from_pretrained(
    pretrained_model, safety_checker=None
).to(device)
pipeline    = accelerator.prepare(pipeline)
pipeline.safety_checker = dummy_safety_checker

# Custom seed schedule for SD XL to improve diversity
# (Some seeds produce similar images with XL, so we remap a few)
SEED_REMAP = {1: 13, 2: 16, 4: 21, 7: 28}
SEED_REMAP_INV = {v: k for k, v in SEED_REMAP.items()}

# ===========================================================================
# Image Generation Loop
# ===========================================================================
for class_label, class_name in enumerate(all_names):
    img_dir = os.path.join(args.gen_root_path, exp_identifier, class_name)
    Path(img_dir).mkdir(parents=True, exist_ok=True)

    # Skip if already generated
    if os.listdir(img_dir):
        print(f"[{class_label}/{len(all_names)}] '{class_name}': already done. Skipping.")
        continue

    print(f"=====> [{class_label}/{len(all_names)}] Generating '{class_name}'")
    prompt = f"A photo of a {class_name}."

    for seed in range(args.Ngen):
        # Remap problematic seeds for diversity (SD XL specific)
        actual_seed = SEED_REMAP.get(seed, seed)

        generator = torch.Generator(device=device)
        generator.manual_seed(actual_seed)

        with accelerator.autocast():
            if args.sd_xl:
                image_out = pipeline(
                    prompt,
                    num_images_per_prompt=1,
                    generator=generator,
                    num_inference_steps=25,
                    height=512,
                    width=512,
                    guidance_scale=7.5,
                    output_type="pt"
                ).images[0]
            else:
                # SD 2.1 or 1.4: standard generation
                image_out = pipeline(
                    prompt, output_type="pt", generator=generator
                ).images[0]

        # Use original (pre-remap) seed index in the filename for consistency
        display_seed = SEED_REMAP_INV.get(actual_seed, seed)
        img_path = os.path.join(img_dir, f"{display_seed}_{class_name}.jpg")

        to_pil     = T.ToPILImage()
        pil_image  = to_pil(image_out.cpu())
        pil_image.save(img_path, "JPEG")
        print(f"  Saved: {img_path}")
        logger.info(f"Generated: {img_path}")
