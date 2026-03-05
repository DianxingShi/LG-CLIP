"""
prompts_gen.py
--------------
Generate diverse image-generation prompts for each class in a dataset using
the Grok (xAI) API. Prompts are saved to a JSON file for use by llm_sd_gen.py.

Features:
  - Incrementally saves after each class (resume-safe).
  - Skips classes already present in the output JSON.
  - Sorts the final output to match dataset class order.

Usage:
  python prompts_gen.py \\
      --dataset CUB \\
      --output_dir dataset/prompts \\
      --api_key YOUR_XAI_API_KEY \\
      --num_prompts 10
"""

import os
import json
import argparse
import requests
from pathlib import Path

from utils.myDataset import (
    CUBDataset, FLODataset, PETDataset, FOODDataset,
    ImageNetDataset, EUROSATDataset
)


# ===========================================================================
# Prompt Generation via xAI Grok API
# ===========================================================================

def generate_prompts(class_name: str, num_prompts: int, api_key: str) -> list:
    """
    Call the xAI Grok API to generate `num_prompts` visual descriptions
    for the given `class_name`. Each description is intended as an SD prompt.

    Returns a list of prompt strings (one per line from the API response).
    """
    url = "https://api.x.ai/v1/chat/completions"
    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    system_msg = (
        "You are a creative assistant helping to generate detailed visual "
        "descriptions for machine learning image generation tasks."
    )
    user_msg = (
        f"Write {num_prompts} detailed visual descriptions of {class_name}, "
        f"each starting with '{class_name}, which has ...'. "
        "The descriptions are used as Stable Diffusion prompts, so each must "
        "describe clear, unambiguous visual features (color, shape, texture, etc.). "
        "Use precise adjectives that accurately reflect the real appearance. "
        "Example format: 'abyssinian, which has a sleek golden-tan coat, slender frame, "
        "tall ears, bright amber eyes, and long agile legs.' "
        "Each entry should be roughly the same length. "
        "Output only the descriptions, no numbering, no extra commentary. "
        "Try to vary viewpoints and do not repeat similar descriptions. "
        "Do not include special characters like ** that could break JSON."
    )
    payload = {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
        "model":       "grok-beta",
        "stream":      False,
        "temperature": 0.7,
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        content = response.json()["choices"][0]["message"]["content"]
        # Split by newline; filter empty lines
        prompts = [line.strip() for line in content.split('\n') if line.strip()]
        return prompts
    else:
        print(f"[API Error] class='{class_name}' "
              f"status={response.status_code}: {response.text}")
        return [f"Error generating prompt for {class_name}"]


# ===========================================================================
# Argument Parsing
# ===========================================================================

projectPath = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(
    description="Generate SD prompts for each dataset class via the xAI Grok API."
)
# --- Dataset ---
parser.add_argument('--dataset',     default='CUB',
                    help='Dataset name: CUB | FLO | PET | FOOD | ImageNet | EUROSAT')
parser.add_argument('--image_root',  default=os.path.join(projectPath, 'dataset'),
                    help='Root directory containing all datasets (default: ./dataset)')
# --- Output ---
parser.add_argument('--output_dir',  default=os.path.join(projectPath, 'dataset', 'prompts'),
                    help='Directory to save the generated prompts JSON (default: ./dataset/prompts)')
# --- API ---
parser.add_argument('--api_key',     default='',
                    help='xAI API key (required)')
parser.add_argument('--num_prompts', default=10, type=int,
                    help='Number of prompts to generate per class (default: 10)')
args = parser.parse_args()


def main():
    # --- Load dataset class names ---
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

    # --- Prepare output file (resume-safe) ---
    output_dir  = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"prompts_for_{args.dataset}.json"

    # Load existing prompts if the file already exists (allow resumption)
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            prompts_dict = json.load(f)
    else:
        prompts_dict = {}

    # Only process classes not yet in the file
    missing_classes = [name for name in all_names if name not in prompts_dict]
    print(f"Total classes: {len(all_names)}, missing: {len(missing_classes)}")

    # --- Generate prompts incrementally ---
    for class_name in missing_classes:
        print(f"  Generating prompts for: {class_name}")
        prompts = generate_prompts(class_name, args.num_prompts, args.api_key)
        prompts_dict[class_name] = prompts

        # Save after each class to avoid losing progress on failure
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(prompts_dict, f, ensure_ascii=False, indent=4)

    # --- Re-sort to match dataset class order ---
    sorted_prompts_dict = {name: prompts_dict.get(name, []) for name in all_names}
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_prompts_dict, f, ensure_ascii=False, indent=4)

    print(f"Prompts saved to: {output_file}")


if __name__ == "__main__":
    main()
