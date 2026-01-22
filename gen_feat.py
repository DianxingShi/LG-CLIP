import os
import re
import h5py
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import json
from tqdm import tqdm
from PIL import Image

import clip
from utils.myDataset import *
from utils.helper_func import *


def get_textEmbedding(classnames, clip_model, args, norm=True):
    """
        - CLIP text embeddings
        - Note: features are normalized
    """
    with torch.no_grad():
        classnames = [classname.replace('_', ' ') for classname in classnames]
        if args.dataset == "AWA2":
            classnames = [classname.replace('+', ' ') for classname in classnames]
        text_descriptions = [f"A photo of a {classname}." for classname in classnames]
        text_tokens = clip.tokenize(text_descriptions, context_length=77).to(args.device)
        text_features = clip_model.encode_text(text_tokens).float().to(args.device)
        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features


def get_visualEmbedding(clip_model, dataframe, device, transform=None):
    """
        - CLIP visual embeddings
        - Note: features are normalized
    """
    with torch.no_grad():
        features = []
        labels = []
        progress = tqdm(total=len(dataframe), ncols=100)
        for img_path, label in dataframe:
            progress.update(1)
            img = Image.open((img_path)).convert('RGB')
            if transform is not None:
                img = transform(img)
            img = img.unsqueeze(0).to(device)
            feature = clip_model.encode_image(img).float().to(device)
            feature /= feature.norm(dim=-1, keepdim=True)
            features.append(feature.cpu())
            labels.append(label)
        progress.close()
    features = np.concatenate(features, axis=0)
    labels = np.array(labels)
    return features, labels


projectPath = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description="")
# -------------------- Path config --------------------#
parser.add_argument('--dataset', default='ImageNet', help='dataset: CUB')
parser.add_argument('--image_root', default=projectPath + '/dataset', help='Path to image root')
parser.add_argument('--gen_root_path', default=projectPath + '/dataset/LLM_SD_gen')                        
# -------------------- other config --------------------#
parser.add_argument('--sd_2_1', default=True, action="store_true", help='SD version')
parser.add_argument('--sd_xl', default=False, action="store_true", help='Use SD XL model')
parser.add_argument('--Ngen', default=10, type=int, help='number of generated images')
parser.add_argument('--backbone', default='RN50', help='CLIP backbone')
parser.add_argument('--LLM', default='', help='LLM_ or None')
parser.add_argument('--seed', default=2024, type=int, help='seed for reproducibility')
parser.add_argument('--device', default='cuda:0', help='cpu/cuda:x')
args = parser.parse_args()
# ======================================== Set random seed ======================================== #
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cudnn.benchmark = True
# ======================================== CLIP ======================================== #
clip_model, preprocess = clip.load(args.backbone, device=args.device)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False
# ======================================== Prepare dataset ======================================== #
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
# ======================================== CLIP features ======================================== #
model_version = "XL" if args.sd_xl else "2.1" if args.sd_2_1 else "1.4"
exp_identifier = f"{args.LLM}SD_{model_version}_{args.dataset}_10"                      
model_name = args.backbone.replace("-", "").replace("/", "")
CLIP_feature_gen_path = args.image_root + f"/{args.dataset}/{args.LLM}CLIP_{model_name}_feature_gen{args.Ngen}.hdf5"                       

if os.path.exists(CLIP_feature_gen_path):
    print(" ==> Load existing feature.")
    hf = h5py.File(CLIP_feature_gen_path, 'r')
    gen_f, gen_l = np.array(hf.get('gen_f')), np.array(hf.get('gen_l'))
    all_embeddings = np.array(hf.get('all_embeddings'))
    print(" ====> Feature loaded.")
    gen_f = torch.from_numpy(gen_f).float().to(args.device)
    gen_l = torch.from_numpy(gen_l).float().to(args.device)
    all_embeddings = torch.from_numpy(all_embeddings).to(args.device)
else:
    print(" ==> Extract feature now.")
    # -------------------- Textual features --------------------#
    print(" ====> Getting textual features from CLIP's text Encoder.")
    all_embeddings = get_textEmbedding(all_names, clip_model, args)
    print("Text embeddings shape: ", all_embeddings.shape)
    # -------------------- Visual features --------------------#
    if args.LLM=="LLM_":
        grp=args.gen_root_path
    else:
        grp=projectPath + '/dataset/SD_gen'
    gen_files, gen_labels = [], []
    if args.Ngen == 10:
        # 原始读取方式
        for idx, name in enumerate(all_names):
            imgDir = f"{grp}/{exp_identifier}/{name}"
            for this_img in os.listdir(imgDir):
                filePath = os.path.join(imgDir, this_img)
                gen_files.append(filePath)
            gen_labels += [idx] * args.Ngen
    else:
        # 从 JSON 文件读取
        json_path = os.path.join(
            args.image_root,
            f"{args.dataset}/json/{args.LLM}SD_{model_version}_{args.dataset}_{model_name}_{args.Ngen}.json"                    
        )
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file {json_path} not found!")
        with open(json_path, 'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)
            for idx, name in enumerate(all_names):
                for entry in json_data[name]:
                    gen_files.append(entry['image'])
                    gen_labels.append(idx)

    assert len(gen_files) == len(all_names) * args.Ngen
    gen_labels = np.array(gen_labels)
    gendf = list(zip(gen_files, gen_labels))
    print(" ====> Getting visual features from CLIP's visual Encoder.")
    gen_f, gen_l = get_visualEmbedding(clip_model, gendf, args.device, transform=preprocess)
    gen_f = gen_f.reshape(len(all_names), args.Ngen, gen_f.shape[-1])
    gen_l = gen_l.reshape(len(all_names), args.Ngen, 1)
    print("Gen embeddings shape: ", gen_f.shape)

    f = h5py.File(CLIP_feature_gen_path, "w")
    f.create_dataset('gen_f', data=gen_f, compression="gzip")
    f.create_dataset('gen_l', data=gen_l, compression="gzip")
    f.create_dataset('all_embeddings', data=all_embeddings.cpu(), compression="gzip")
    f.close()

    print(" ====> Feature saved.")
    gen_f = torch.from_numpy(gen_f).float().to(args.device)
    gen_l = torch.from_numpy(gen_l).float().to(args.device)
# ======================================== CLIP classifier ======================================== #
gen_f = gen_f.view(-1, gen_f.size(-1))
gen_l = gen_l.view(-1)
simi_scores = torch.matmul(gen_f, all_embeddings.T)
predicted_classes = torch.argmax(simi_scores, dim=1)
correct = torch.sum(predicted_classes == gen_l)
acc = correct.item() / len(gen_l)
print("[{}]: Gen Acc={:.2f}%".format(args.backbone, acc * 100))
