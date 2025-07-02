import os
import re
import torch
import random
import warnings
import argparse
import numpy as np
from PIL import Image
import torch.nn as nn
from pathlib import Path
import torch.backends.cudnn as cudnn
from diffusers import StableDiffusionPipeline
from utils.myDataset import *
from utils.helper_func import *
warnings.filterwarnings('ignore')

def dummy_safety_checker(images, clip_input=None):
    """
    A dummy safety checker that bypasses the NSFW content filter.
    """
    return images, False

projectPath = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description="")
# -------------------- Path config --------------------#
parser.add_argument('--dataset', default='ImageNet', help='dataset: AWA2/CUB/SUN')
parser.add_argument('--image_root', default=projectPath + '/dataset', help='Path to image root')
parser.add_argument('--gen_root_path', default=projectPath + '/dataset/SD_gen')
# -------------------- other config --------------------#
parser.add_argument('--sd_2_1', default=True, action="store_true", help='SD version')
parser.add_argument('--sd_xl', default=False, action="store_true", help='Use SD XL version')
parser.add_argument('--Ngen', default=10, type=int, help='number of generated images')
parser.add_argument('--seed', default=2024, type=int, help='seed for reproducibility')
parser.add_argument('--device', default='cuda:0', help='cpu/cuda:x')
args = parser.parse_args()
# ======================================== Set random seed ======================================== #
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cudnn.benchmark = True
# ======================================== Dataset ======================================== #
if args.dataset == "CUB":
    mydataset = CUBDataset(args)
elif args.dataset == "FLO":
    mydataset = FLODataset(args)
elif args.dataset == "PET":
    mydataset = PETDataset(args)
elif args.dataset == "FOOD":
    mydataset = FOODDataset(args)
elif args.dataset == "ImageNet":
    mydataset = ImageNetDataset(args)
elif args.dataset == "EUROSAT":
    mydataset = EUROSATDataset(args)
else:
    raise ValueError("UnKnown Dataset!")
all_names = mydataset.all_names
# ======================================== Main Pipeline ======================================== #
model_version = "XL" if args.sd_xl else "2.1" if args.sd_2_1 else "1.4"
exp_identifier = f"SD_{model_version}_{args.dataset}_{args.Ngen}"
print(f"==> Start Generation: {exp_identifier}")
# ---------- Run Log ----------
import logging
eval_log_path = f"{args.gen_root_path}/{exp_identifier}"
Path(eval_log_path).mkdir(parents=True, exist_ok=True)
log = eval_log_path + "/sd_gen_log.txt"
logging.basicConfig(format='%(message)s', level=logging.INFO,
                    filename=log,
                    filemode='w')
logger = logging.getLogger(__name__)
argsDict = args.__dict__
for eachArg, value in argsDict.items():
    logger.info(eachArg + ':' + str(value))
logger.info("="*50)

for class_label, class_name in enumerate(all_names):
    print(f"====> Start generating '{class_label}:{class_name}'")
    logger.info(f"====> Start generating '{class_label}:{class_name}'")

    img_dir_path = f"{args.gen_root_path}/{exp_identifier}/{class_name}"
    Path(img_dir_path).mkdir(parents=True, exist_ok=True)

    # 检查文件夹中已有的图像数量
    existing_images = [f for f in os.listdir(img_dir_path) if f.endswith('.jpg')]
    num_existing = len(existing_images)

    if num_existing >= args.Ngen:
        print(f"Images already exist at {img_dir_path} ({num_existing}/{args.Ngen})")
        logger.info(f"Images already exist at {img_dir_path} ({num_existing}/{args.Ngen})")
        continue

    # -------------------- Pipeline -------------------- #
    if args.sd_2_1:
        pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-base"
    elif args.sd_xl:
        pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
    else:
        pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"

    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path, safety_checker=None).to(args.device)

    prompt = f"A photo of a {class_name}."
    print(f"Generation of the prompt: '{prompt}'")
    logger.info(f"Generation of the prompt: '{prompt}'")

    # 确定起始序号，继续生成图像
    existing_indices = [int(re.search(r'^(\d+)_', img).group(1)) for img in existing_images if re.match(r'^\d+_', img)]
    start_index = max(existing_indices) + 1 if existing_indices else 0

    for i in range(start_index, args.Ngen):
        generator = torch.Generator(device=args.device)
        generator.manual_seed(i)
        image_out = pipeline(prompt, output_type="pt", generator=generator)[0]
        img_path = f"{img_dir_path}/{i}_{class_name}.jpg"
        numpy_to_pil(
            image_out.permute(0, 2, 3, 1).cpu().detach().numpy())[0].save(img_path, "JPEG")

        print(f"Generated image: {img_path}")
        logger.info(f"Generated image: {img_path}")
