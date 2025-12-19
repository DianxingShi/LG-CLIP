

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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
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
        # prompt ensemble for ImageNet
        text_features = clip_model.encode_text(text_tokens).float().to(args.device)
        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)
        class_embeddings = text_features.to(args.device)
    return class_embeddings


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
parser.add_argument('--dataset', default='FLO', help='dataset: AWA2/CUB/SUN')
parser.add_argument('--image_root', default= projectPath + '/dataset', help='Path to image root')
# -------------------- other config --------------------#
parser.add_argument('--backbone', default='RN50', help='[RN50, ViT-B/32, ViT-B/16, ViT-L/14]')
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
# print(all_names)
# ======================================== CLIP features ======================================== #
model_name = args.backbone.replace("-", "").replace("/", "")
CLIP_feature_path = args.image_root + f"/{args.dataset}/CLIP_{model_name}_feature_multi_scale.hdf5"
if os.path.exists(CLIP_feature_path):
    print(" ==> Load existing feature.")
    hf = h5py.File(CLIP_feature_path, 'r')
    test_f, test_l = np.array(hf.get('test_f')), np.array(hf.get('test_l'))
    all_embeddings = np.array(hf.get('all_embeddings'))
    print(" ====> Feature loaded.")
    test_f = torch.from_numpy(test_f).float().to(args.device)
    test_l = torch.from_numpy(test_l).float().to(args.device)
    all_embeddings = torch.from_numpy(all_embeddings).to(args.device)
else:
    print(" ==> Extract feature now.")
    # -------------------- Textual features --------------------#
    print(" ====> Getting textual features from CLIP's text Encoder.")
    all_embeddings = get_textEmbedding(all_names, clip_model, args)
    print("Text embeddings shape: ", all_embeddings.shape)
    # -------------------- Visual features --------------------#
    traindf = list(zip(mydataset.train_files, mydataset.train_labels.numpy()))
    testdf = list(zip(mydataset.test_files, mydataset.test_labels.numpy()))
    print(" ====> Getting visual features from CLIP's visual Encoder.")
    train_f, train_l = get_visualEmbedding(clip_model, traindf, args.device, transform = preprocess)
    test_f, test_l = get_visualEmbedding(clip_model, testdf, args.device, transform = preprocess)
    print("Train embeddings shape: ", train_f.shape)
    print("Test embeddings shape: ", test_f.shape)
    f = h5py.File(CLIP_feature_path, "w")
    f.create_dataset('train_f', data = train_f, compression = "gzip")
    f.create_dataset('train_l', data = train_l, compression = "gzip")
    f.create_dataset('test_f', data = test_f, compression = "gzip")
    f.create_dataset('test_l', data = test_l, compression = "gzip")
    f.create_dataset('all_embeddings', data = all_embeddings.cpu(), compression = "gzip")
    f.close()
    print(" ====> Feature saved.")
    test_f = torch.from_numpy(test_f).float().to(args.device)
    test_l = torch.from_numpy(test_l).float().to(args.device)
# ======================================== CLIP classifier ======================================== #
simi_scores = torch.matmul(test_f, all_embeddings.T)
predicted_classes = torch.argmax(simi_scores, dim=1)
correct = torch.sum(predicted_classes == test_l)
Acc = correct.item() / len(test_l)
print("[{}]: Acc = {:.2f}%".format(args.backbone, Acc * 100))
# mapped_results = []
# for image_file, label in zip(mydataset.image_files, mydataset.labels.numpy()):
#     class_name = mydataset.all_names[label - 1] 
#     mapped_results.append(f" {class_name}")


# output_file = "imagenet_mapped_results.txt"
# with open(output_file, "w") as f:
#     for result in mapped_results:
#         f.write(result + "\n")

# print(f"Mapping results saved to {output_file}")
