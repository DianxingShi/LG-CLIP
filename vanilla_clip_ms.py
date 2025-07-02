import os
import re
import h5py
import torch
import random
import argparse
import numpy as np
import torch.backends.cudnn as cudnn
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
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
            img = Image.open(img_path).convert('RGB')
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
import re
def load_feature_and_label(feature_dir, scale_idx,bc):
    """
    Load features and labels from the HDF5 file for a specific scale index.
    """
    # file_path = os.path.join(feature_dir, f"LLM_CLIP_{bc}_feature_gen5_scale{scale_idx}.hdf5")   ##生成集多尺度目录
    file_path = os.path.join(feature_dir, f"CLIP_{bc}_feature_scale{scale_idx}.hdf5")      ##测试集多尺度目录
    with h5py.File(file_path, 'r') as f:
        features = torch.from_numpy(np.array(f['test_f'])).float()
        labels = torch.from_numpy(np.array(f['test_l'])).float()
        all_embedding = np.array(f.get('all_embeddings'))
    return features, labels,all_embedding
    
def save_feature_and_label(features, labels,all_embedding,save_path):
    """
    Save features and labels into an HDF5 file.
    """
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('test_f', data=features.cpu().numpy(), compression="gzip")
        f.create_dataset('test_l', data=labels.cpu().numpy(), compression="gzip")
        f.create_dataset('all_embeddings', data=all_embedding, compression="gzip")
def combine_multi_scale_features(base_dir, save_path, scales=10):
    """
    Combine multi-scale features and labels into one aggregated file.
    """
    features_all = None
    labels_all = None

    for scale in range(1, scales + 1):
        print(f"Processing scale {scale}...")

        # Load the current scale's features and labels
        features, labels,all_embedding= load_feature_and_label(base_dir, scale,model_name)

        # Calculate the weighting ratio
        ratio = scale / 55.0

        # Aggregate features
        if features_all is None:
            features_all = features * ratio
            labels_all = labels  # Labels are not scaled, use the first scale's labels
        else:
            features_all += features * ratio

    # Normalize the aggregated features
    features_all /= features_all.norm(dim=-1, keepdim=True)
    all_embedding = all_embedding
    # Save the combined features and labels
    save_feature_and_label(features_all, labels_all,all_embedding ,save_path)

projectPath = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description="Multi-scale feature extraction with CLIP")
# -------------------- Path config --------------------#
parser.add_argument('--dataset', default='ImageNet', help='dataset: AWA2/CUB/SUN')
parser.add_argument('--image_root', default=projectPath + '/dataset', help='Path to image root')
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
    raise ValueError("Unknown Dataset!")
all_names = mydataset.all_names
# ======================================== CLIP features ======================================== #
model_name = args.backbone.replace("-", "").replace("/", "")
multi_scale_dir = os.path.join(args.image_root, args.dataset, "multi_scale")
os.makedirs(multi_scale_dir, exist_ok=True)

# Textual features (once for all scales)
all_embeddings = get_textEmbedding(all_names, clip_model, args)
print("Text embeddings shape: ", all_embeddings.shape)

for scale in range(1, 11):  # Multi-scale feature extraction (10 rounds)
    scale_factor = scale / 10
    print(f"Processing scale factor: {scale_factor}")
    CLIP_feature_path = os.path.join(multi_scale_dir, f"CLIP_{model_name}_feature_scale{scale}.hdf5")
    if os.path.exists(CLIP_feature_path):
        print(f" ==> Load existing feature for scale {scale}.")
        continue

    # Apply multi-scale transformation to preprocess images
    multi_scale_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=224, scale=(scale_factor, scale_factor), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    ])

    # Extract visual features
    testdf = list(zip(mydataset.test_files, mydataset.test_labels.numpy()))
    print(f" ====> Getting visual features from CLIP's visual Encoder for scale {scale}.")
    test_f, test_l = get_visualEmbedding(clip_model, testdf, args.device, transform=multi_scale_transform)
    print(f"Test embeddings shape for scale {scale}: ", test_f.shape)
    # Save features
    with h5py.File(CLIP_feature_path, "w") as f:
        f.create_dataset('test_f', data=test_f, compression="gzip")
        f.create_dataset('test_l', data=test_l, compression="gzip")
        f.create_dataset('all_embeddings', data=all_embeddings.cpu(), compression="gzip")
        f.close()
        test_f = torch.from_numpy(test_f).float().to(args.device)
        test_l = torch.from_numpy(test_l).float().to(args.device)
    print(f" ====> Feature saved for scale {scale}.")

    simi_scores = torch.matmul(test_f, all_embeddings.T)
    predicted_classes = torch.argmax(simi_scores, dim=1)
    correct = torch.sum(predicted_classes == test_l)
    Acc = correct.item() / len(test_l)
    print("[{}]: Acc{} = {:.2f}%".format(args.backbone,scale,Acc * 100))

# Configuration
dataset_name = args.dataset
model_name = args.backbone.replace("-", "").replace("/", "")
# gen5t_model_name = re.sub(r"[A-Za-z]", "", args.backbone.replace("-", "").replace("/", ""))
base_real_feature_dir = f"D:\\PYproject\\clipI\\dataset\\{dataset_name}\\multi_scale"  ##测试集特征目录
save_real_feature_path = f"D:\\PYproject\\clipI\\dataset\\{dataset_name}\\CLIP_{model_name}_feature_ms.hdf5"   ##测试集最终特征保存路径
# Combine features
combine_multi_scale_features(base_real_feature_dir, save_real_feature_path)
print("Combined multi-scale features and labels saved successfully!")