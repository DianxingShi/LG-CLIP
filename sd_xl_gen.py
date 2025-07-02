"""
    - Generating Images with Stable Diffusion
"""
import os
import torch
import random
import warnings
import argparse
import torchvision.transforms as T
from pathlib import Path
from diffusers import StableDiffusionPipeline
from diffusers import StableDiffusionXLPipeline
from diffusers import AutoPipelineForText2Image
from accelerate import Accelerator
from utils.myDataset import *
from utils.helper_func import *
import logging  # Add logging
warnings.filterwarnings('ignore')

os.environ["XFORMERS_OFFLOAD"] = "1"  # Set to 1 to offload to CPU
os.environ["XFORMERS_THRESHOLD"] = "1"  # Set the threshold for using 
def dummy_safety_checker(images, clip_input=None):
    """ 
    A dummy safety checker that bypasses the NSFW content filter.
    """
    return images, False




# -------------------- Argument Parser -------------------- #
projectPath = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description="Stable Diffusion Generator")
parser.add_argument('--dataset', default='FLO', help='dataset: AWA2/CUB/SUN')
parser.add_argument('--image_root', default=projectPath + '/dataset', help='Path to image root')
parser.add_argument('--gen_root_path', default=projectPath + '/dataset/SD_gen')
parser.add_argument('--sd_2_1', default=False, action="store_true", help='Use SD 2.1 model')
parser.add_argument('--sd_xl', default=True, action="store_true", help='Use SD XL model')
parser.add_argument('--Ngen', default=10, type=int, help='Number of generated images per class')
parser.add_argument('--seed', default=2024, type=int, help='Seed for reproducibility')
parser.add_argument('--device', default='cuda:0', help='Device to use (e.g., cpu or cuda:x)')
args = parser.parse_args()

# -------------------- Set Random Seed -------------------- #
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# -------------------- Dataset -------------------- #
if args.dataset == "CUB":
    mydataset = CUBDataset(args)
elif args.dataset == "FLO":
    mydataset = FLODataset(args)
elif args.dataset == "PET":
    mydataset = PETDataset(args)
elif args.dataset == "FOOD":
    mydataset = FOODDataset(args)
else: 
    raise ValueError("Unknown Dataset!")
all_names = mydataset.all_names

# -------------------- Define Model Version -------------------- #
model_version = "XL" if args.sd_xl else "2.1" if args.sd_2_1 else "1.4"
exp_identifier = f"SD_{model_version}_{args.dataset}_{args.Ngen}"

# -------------------- Logging -------------------- #
eval_log_path = f"{args.gen_root_path}/{exp_identifier}"
Path(eval_log_path).mkdir(parents=True, exist_ok=True)
log = eval_log_path + "/sd_gen_log.txt"
logging.basicConfig(filename=log, filemode='w', level=logging.INFO)

# -------------------- Load Model -------------------- #
if args.sd_xl:
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-xl-base-1.0"
elif args.sd_2_1:
    pretrained_model_name_or_path = "stabilityai/stable-diffusion-2-1-base"
else:
    pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"

accelerator = Accelerator()  # 支持 FP16
# mixed_precision="fp16"
device = accelerator.device
pipeline = AutoPipelineForText2Image.from_pretrained(pretrained_model_name_or_path,safety_checker=None).to(device)
pipeline = accelerator.prepare(pipeline)
# torch.float16 if torch.cuda.is_available() else 
pipeline.safety_checker = dummy_safety_checker

# -------------------- Generate Images -------------------- #
for class_label, class_name in enumerate(all_names):
    print(f"====> Generating images for '{class_name}'")
    img_dir_path = f"{args.gen_root_path}/{exp_identifier}/{class_name}"
    Path(img_dir_path).mkdir(parents=True, exist_ok=True)

    if os.listdir(img_dir_path):
        print(f"Images already exist for {class_name}, skipping...")
        continue

    prompt = f"A photo of a {class_name}."

    for seed in range(args.Ngen):
        if seed==1:
            seed=13
        elif seed==2:
            seed=16
        elif seed==4:
            seed=21
        elif seed==7:
            seed=28
        
        
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        with accelerator.autocast(): 
            if args.sd_xl:
          
                    image_out = pipeline(
                        prompt,
                        num_images_per_prompt=1,  # Number of images per prompt
                        generator=generator,
                        num_inference_steps=25,  # Set the number of inference steps as needed
                        height=512,  # Image height
                        width=512,  # Image width
                        guidance_scale=7.5,  
                        output_type="pt"
                    ).images[0]
                # print(image_out)
            else:
            # SD 2.1 or 1.4: Use regular prompt without special handling
                image_out = pipeline(prompt, output_type="pt",generator=generator).images[0]
        if seed==13:
            seed=1
        elif seed==16:
            seed=2
        elif seed==21:
            seed=4
        elif seed==28:
            seed=7
        img_path = f"{img_dir_path}/{seed}_{class_name}.jpg"
        to_pil = T.ToPILImage()
        pil_image = to_pil(image_out.cpu())  # 直接将 [C, H, W] 张量转换为 PIL 图像

# 保存为 JPEG 文件
        pil_image.save(img_path, "JPEG") 