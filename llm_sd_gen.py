import os
import re
import torch
import random
import logging
import argparse
import numpy as np
from pathlib import Path
from diffusers import StableDiffusionPipeline
from diffusers import AutoPipelineForText2Image
import torchvision.transforms as T
from utils.helper_func import *
from accelerate import Accelerator
from PIL import Image  # 这里是导入 PIL 的 Image 模块
import json

# 加载 JSON 文件
def load_class_prompts(json_file_path):
    with open(json_file_path, 'r') as file:
        class_prompts = json.load(file)
    return class_prompts

# 设置参数
projectPath = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description="Image Generation with Stable Diffusion")
# -------------------- Path config --------------------#
parser.add_argument('--dataset', default='CUB', help='dataset: PET')
parser.add_argument('--image_root', default= projectPath + '/dataset', help='Path to image root')
parser.add_argument('--gen_root_path', default=projectPath + '/dataset/LLM_SD_gen')
parser.add_argument('--sd_2_1', default=True, action="store_true", help='SD version')
parser.add_argument('--sd_xl', default=False, action="store_true", help='Use SD XL model')
parser.add_argument('--Ngen', default=10, type=int, help='number of generated images')
parser.add_argument('--seed', default=2024, type=int, help='seed for reproducibility')
parser.add_argument('--device', default='cuda:0', help='cpu/cuda:x')
parser.add_argument('--json_file', default=r"D:\PYproject\clipI\dataset\prompts\CUB.json", help='Path to the JSON file with class prompts')
args = parser.parse_args()

# ======================================== Set random seed ======================================== #
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# ======================================== Dataset ======================================== #
# Only load PET dataset and ignore othersyv
# if args.dataset == "PET":
#     # 加载从JSON文件生成的提示词数据
class_prompts = load_class_prompts(args.json_file)
# elif args.dataset == "FLO":   
#     # 加载从JSON文件生成的提示词数据
#     class_prompts = load_class_prompts(args.json_file)
# else:
#     print(f"Dataset {args.dataset} is not supported. Only 'PET' dataset will be used.")
#     exit()

# ======================================== Main Pipeline ======================================== #
model_version = "XL" if args.sd_xl else "2.1" if args.sd_2_1 else "1.4"
exp_identifier = f"LLM_SD_{model_version}_{args.dataset}_{args.Ngen}"
print(f"==> Start Generation: {exp_identifier}")

# Set up logging
eval_log_path = f"{args.gen_root_path}/{exp_identifier}"
Path(eval_log_path).mkdir(parents=True, exist_ok=True)
log = eval_log_path + "/llm_sd_gen_log.txt"
logging.basicConfig(format='%(message)s', level=logging.INFO, filename=log, filemode='w')
logger = logging.getLogger(__name__)

# Log arguments
argsDict = args.__dict__
for eachArg, value in argsDict.items():
    logger.info(eachArg + ':' + str(value))
logger.info("="*50)

# 生成图像
for class_name, prompts in class_prompts.items():
    # 每个类使用第一个提示词生成图像
    prompt = prompts[0]
    print(f"====> Start generating images for '{class_name}' with prompt: '{prompt}'")
    logger.info(f"====> Start generating images for '{class_name}' with prompt: '{prompt}'")

    # 创建保存图像的目录
    img_dir_path = f"{args.gen_root_path}/{exp_identifier}/{class_name}"
    Path(img_dir_path).mkdir(parents=True, exist_ok=True)

    existing_images = [f for f in os.listdir(img_dir_path) if f.endswith('.jpg') or f.endswith('.png')]
    existing_count = len(existing_images)
# 确定起始序号：找到当前目录下的最大序号
    if existing_count >= args.Ngen:  # 文件夹已满，跳过生成
        continue
    max_index = 0
    if existing_images:
    # 提取已有图片中的序号部分
        indices = [int(re.search(r'^(\d+)_', img).group(1)) for img in existing_images if re.search(r'^(\d+)_', img)]
        max_index = max(indices) + 1  # 序号从最大值+1开始
    else:
        max_index = 0  # 如果没有图片，序号从0开始

    # -------------------- Pipeline -------------------- #
    if args.sd_xl:
        pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
    elif args.sd_2_1:
        pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-base"
    else:
        pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
    
    # 加载Stable Diffusion模型
    accelerator = Accelerator()  # 支持 FP16
# mixed_precision="fp16"
    device = accelerator.device
    pipeline = AutoPipelineForText2Image.from_pretrained(pretrained_model_name_or_path).to(device)
    pipeline = accelerator.prepare(pipeline)

    for seed in range(existing_count, args.Ngen):  # 从现有数量继续生成
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)  # 使用当前种子生成器

    # 使用序号生成唯一文件名
        img_name = f"{max_index + (seed - existing_count)}_{class_name}.jpg"
        img_path = os.path.join(img_dir_path, img_name)

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
            image_out = pipeline(prompt, output_type="pt", generator=generator).images[0]

    # 将生成的图像保存为 JPEG
        to_pil = T.ToPILImage()
        pil_image = to_pil(image_out.cpu())
        pil_image.save(img_path, "JPEG")

    # 打印日志
        print(f"Generated image: {img_path}")
        logger.info(f"Generated image: {img_path}")

        