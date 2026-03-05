"""
llm_sd_gen.py
-------------
Generate images using Stable Diffusion guided by LLM-generated prompts.
Prompts are loaded from a JSON file produced by prompts_gen.py.

Supports:
  - SD v1.4  (CompVis/stable-diffusion-v1-4)
  - SD v2.1  (stabilityai/stable-diffusion-2-1-base)  [default]
  - SD XL    (stabilityai/stable-diffusion-xl-base-1.0)

Output structure:
  <gen_root_path>/LLM_SD_<version>_<dataset>_<Ngen>/<class_name>/<idx>_<class_name>.jpg

Resumable: existing images are detected and skipped.

Usage:
  python llm_sd_gen.py \\
      --dataset CUB \\
      --json_file dataset/prompts/prompts_for_CUB.json \\
      --Ngen 10 [--sd_2_1 | --sd_xl] [--device cuda:0]
"""

import os
import re
import json
import logging
import argparse
import torch
import random
import numpy as np
from pathlib import Path
import torchvision.transforms as T
from diffusers import AutoPipelineForText2Image
from accelerate import Accelerator
from PIL import Image


def load_class_prompts(json_file_path: str) -> dict:
    """Load class-to-prompts mapping from a JSON file."""
    with open(json_file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ===========================================================================
# Argument Parsing
# ===========================================================================

projectPath = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(
    description="Generate images with Stable Diffusion using LLM-enriched prompts."
)
# --- Dataset & Paths ---
parser.add_argument('--dataset',       default='CUB',
                    help='Dataset name (used for experiment naming only)')
parser.add_argument('--image_root',    default=os.path.join(projectPath, 'dataset'),
                    help='Root directory of datasets (default: ./dataset)')
parser.add_argument('--gen_root_path', default=os.path.join(projectPath, 'dataset', 'LLM_SD_gen'),
                    help='Output directory for LLM+SD generated images (default: ./dataset/LLM_SD_gen)')
parser.add_argument('--json_file',     default=os.path.join(projectPath, 'dataset', 'prompts', 'prompts_for_CUB.json'),
                    help='Path to the JSON file with class prompts (from prompts_gen.py)')
# --- SD Version ---
parser.add_argument('--sd_2_1', default=True,  action="store_true",
                    help='Use SD v2.1 (default)')
parser.add_argument('--sd_xl',  default=False, action="store_true",
                    help='Use SD XL (overrides --sd_2_1)')
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
# Reproducibility
# ===========================================================================
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# ===========================================================================
# Load Prompts
# ===========================================================================
class_prompts = load_class_prompts(args.json_file)

# ===========================================================================
# Experiment Setup & Logging
# ===========================================================================
model_version  = "XL" if args.sd_xl else "2.1" if args.sd_2_1 else "1.4"
exp_identifier = f"LLM_SD_{model_version}_{args.dataset}_{args.Ngen}"
print(f"==> Starting generation: {exp_identifier}")

eval_log_path = os.path.join(args.gen_root_path, exp_identifier)
Path(eval_log_path).mkdir(parents=True, exist_ok=True)
log_file = os.path.join(eval_log_path, "llm_sd_gen_log.txt")
logging.basicConfig(
    format='%(message)s', level=logging.INFO,
    filename=log_file, filemode='w'
)
logger = logging.getLogger(__name__)
for k, v in args.__dict__.items():
    logger.info(f"{k}: {v}")
logger.info("=" * 50)

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
pipeline    = AutoPipelineForText2Image.from_pretrained(pretrained_model).to(device)
pipeline    = accelerator.prepare(pipeline)

# ===========================================================================
# Image Generation Loop
# ===========================================================================
for class_name, prompts in class_prompts.items():
    # Use the first prompt from the LLM-generated list for this class
    prompt = prompts[0] if prompts else f"A photo of a {class_name}."
    print(f"=====> Generating '{class_name}' | Prompt: '{prompt[:80]}...'")
    logger.info(f"=====> '{class_name}' | prompt: '{prompt}'")

    img_dir = os.path.join(args.gen_root_path, exp_identifier, class_name)
    Path(img_dir).mkdir(parents=True, exist_ok=True)

    # Count existing images (resume support)
    existing_images = [
        f for f in os.listdir(img_dir) if f.endswith('.jpg') or f.endswith('.png')
    ]
    existing_count = len(existing_images)
    if existing_count >= args.Ngen:
        print(f"  Already done ({existing_count}/{args.Ngen}). Skipping.")
        continue

    # Determine the next image index
    if existing_images:
        indices   = [
            int(re.search(r'^(\d+)_', img).group(1))
            for img in existing_images
            if re.search(r'^(\d+)_', img)
        ]
        max_index = max(indices) + 1
    else:
        max_index = 0

    # Generate remaining images
    for seed in range(existing_count, args.Ngen):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        img_name = f"{max_index + (seed - existing_count)}_{class_name}.jpg"
        img_path = os.path.join(img_dir, img_name)

        if args.sd_xl:
            with accelerator.autocast():
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
            # SD 2.1 or 1.4
            image_out = pipeline(prompt, output_type="pt", generator=generator).images[0]

        to_pil = T.ToPILImage()
        to_pil(image_out.cpu()).save(img_path, "JPEG")
        print(f"  Saved: {img_path}")
        logger.info(f"Generated: {img_path}")
