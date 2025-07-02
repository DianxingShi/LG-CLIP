"""
    - Classification with Generated Images (contrastive learning)
"""

import os
import re
import h5py
import torch
import random
import argparse
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import clip
from utils.myDataset import *
from utils.helper_func import *


projectPath = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description="")
# -------------------- Path config --------------------#
parser.add_argument('--dataset', default='EUROSAT', help='dataset: CUB')
parser.add_argument('--image_root', default= projectPath + '/dataset', help='Path to image root')
# parser.add_argument('--gen_root_path', default = projectPath + '/dataset/SD_gen')
# -------------------- other config --------------------#
parser.add_argument('--sd_2_1', default=True, action="store_true", help='SD version')
# parser.add_argument('--sd_xl', default=False, action="store_true", help='Use SD XL version')
parser.add_argument('--Ngen', default=5, type=int, help='number of generated images')
parser.add_argument('--LLM', default='', help='LLM_ or None')
parser.add_argument('--ms1', default='', help='_ms or None')
parser.add_argument('--ms2', default='', help='_ms or None')
parser.add_argument('--backbone', default='RN50', help='CLIP backbone')
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
model_name = args.backbone.replace("-", "").replace("/", "")
CLIP_feature_path = args.image_root + f"/{args.dataset}/CLIP_{model_name}_feature{args.ms1}.hdf5"
CLIP_feature_gen_path = args.image_root + f"/{args.dataset}/{args.LLM}CLIP_{model_name}_feature_gen{args.Ngen}{args.ms2}.hdf5"                        #☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#####################################################################

if os.path.exists(CLIP_feature_path) and os.path.exists(CLIP_feature_gen_path):
    print(" ==> Load existing feature (Real).")
    hf = h5py.File(CLIP_feature_path, 'r')
    test_f, test_l = np.array(hf.get('test_f')), np.array(hf.get('test_l'))
    all_embeddings = np.array(hf.get('all_embeddings'))
    print(" ====> Feature loaded (Real).")
    test_f = torch.from_numpy(test_f).float().to(args.device)
    test_l = torch.from_numpy(test_l).float().to(args.device)
    all_embeddings = torch.from_numpy(all_embeddings).to(args.device)

    print(" ==> Load existing feature (Gen).") 
    hf = h5py.File(CLIP_feature_gen_path, 'r')
    gen_f, gen_l = np.array(hf.get('gen_f')), np.array(hf.get('gen_l'))
    # all_embeddings = np.array(hf.get('all_embeddings'))
    print(" ====> Feature loaded (Gen).")
    gen_f = torch.from_numpy(gen_f).float().to(args.device)
    gen_l = torch.from_numpy(gen_l).float().to(args.device)
else:
    raise ValueError("File Not Found")
# ======================================== CLIP classifier ======================================== #
"""
    - [1] Average gen_f
"""
real_prototypes = all_embeddings
gen_prototypes = torch.mean(gen_f, dim=1)

# U, _, _ = np.linalg.svd(gen_prototypes.cpu())
# _U = U[:, 1:]
# P_text = np.dot(_U, _U.T)
# gen_prototypes = torch.from_numpy(
#     np.dot(P_text, gen_prototypes.cpu())).float().to(args.device)

# import matplotlib.pyplot as plt
# dot_product = torch.matmul(real_prototypes, gen_prototypes.t())
# norm_x = torch.norm(real_prototypes, dim=1, keepdim=True)
# norm_y = torch.norm(gen_prototypes, dim=1, keepdim=True)
# cosine_similarity = dot_product / (norm_x * norm_y.t())
# similarity_matrix = cosine_similarity.cpu().numpy()
# plt.imshow(similarity_matrix, cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.savefig(f'similarity_matrix_{args.dataset}.png')


prototypes = gen_prototypes
# prototypes = (real_prototypes + gen_prototypes) / 2
# prototypes = real_prototypes *31/40 + gen_prototypes *9/40

#####################################################################加权/屏蔽
# mask = torch.ones_like(gen_prototypes)
# mask[24] = 0  # Set the 24th class (peisian) to 0
# masked_gen_prototypes = gen_prototypes * mask
# simi_scores = torch.matmul(test_f, masked_gen_prototypes.T)
# predicted_classes = torch.argmax(simi_scores, dim=1)
# correct = torch.sum(predicted_classes == test_l)
# Acc = correct.item() / len(test_l)
# print("[{}]: Acc = {:.2f}%".format(args.backbone, Acc * 100))

####################################################################
simi_scores = torch.matmul(test_f, prototypes.T)
predicted_classes = torch.argmax(simi_scores, dim=1)
correct = torch.sum(predicted_classes == test_l)
Acc = correct.item() / len(test_l)
print("[{}]: Acc = {:.2f}%".format(args.backbone, Acc * 100))

log_path = os.path.join(projectPath, "metalog.txt")
log_data = (
    f"Dataset: {args.dataset}\n"
    f"Ngen: {args.Ngen}\n"
    f"LLM: {args.LLM}\n"
    f"ms1: {args.ms1}\n"
    f"ms2: {args.ms2}\n"
    f"Backbone: {args.backbone}\n"
    f"CLIP_feature_path: {CLIP_feature_path}\n"
    f"CLIP_feature_gen_path: {CLIP_feature_gen_path}\n"     
    f"Acc: {Acc * 100:.2f}%\n"
    f"{'-'*80}\n"
)

with open(log_path, "a") as log_file:
    log_file.write(log_data)
    print(f"Results saved to {log_path}")