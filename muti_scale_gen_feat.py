import os
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
import torch
import re


def get_textEmbedding(classnames, clip_model, args, norm=True):
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


def load_feature_and_label(feature_dir, scale_idx,bc):
    """
    Load features and labels from the HDF5 file for a specific scale index.
    """
    file_path = os.path.join(feature_dir, f"{args.LLM}CLIP_{bc}_feature_gen{args.Ngen}_scale{scale_idx}.hdf5")   ##生成集多尺度目录#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#####################################################################
    # file_path = os.path.join(feature_dir, f"CLIP_{bc}_scale{scale_idx}.hdf5")      ##测试集多尺度目录
    with h5py.File(file_path, 'r') as f:
        features = torch.from_numpy(np.array(f['gen_f'])).float()
        labels = torch.from_numpy(np.array(f['gen_l'])).float()
    return features, labels

def save_feature_and_label(features, labels, save_path):
    """
    Save features and labels into an HDF5 file.
    """
    with h5py.File(save_path, 'w') as f:
        f.create_dataset('gen_f', data=features.cpu().numpy(), compression="gzip")
        f.create_dataset('gen_l', data=labels.cpu().numpy(), compression="gzip")

def combine_multi_scale_features(base_dir, save_path, scales=10):
    """
    Combine multi-scale features and labels into one aggregated file.
    """
    features_all = None
    labels_all = None

    for scale in range(1, scales + 1):
        print(f"Processing scale {scale}...")

        # Load the current scale's features and labels
        features, labels = load_feature_and_label(base_dir, scale,model_name)

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

    # Save the combined features and labels
    save_feature_and_label(features_all, labels_all, save_path)


projectPath = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description="Extract CLIP features with multi-scale transformations")
# -------------------- Path config --------------------#
parser.add_argument('--dataset', default='PET', help='Dataset: CUB, FLO, etc.')
parser.add_argument('--image_root', default=projectPath + '/dataset', help='Path to dataset root')
parser.add_argument('--gen_root_path', default=projectPath + '/dataset/LLM_SD_gen')                        #☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#####################################################################
# -------------------- Other config --------------------#
parser.add_argument('--sd_2_1', default=True, action="store_true", help='SD version')
parser.add_argument('--sd_xl', default=False, action="store_true", help='Use SD XL model')
parser.add_argument('--Ngen', default=10, type=int, help='Number of generated images per class')
parser.add_argument('--backbone', default='ViT-L/14', help='CLIP backbone')
parser.add_argument('--LLM', default='', help='LLM_ or None')
parser.add_argument('--seed', default=2024, type=int, help='Random seed')
parser.add_argument('--device', default='cuda:0', help='Device: cpu or cuda:x')
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
model_version = "XL" if args.sd_xl else "2.1" if args.sd_2_1 else "1.4"
model_name = args.backbone.replace("-", "").replace("/", "")
gen5t_model_name = re.sub(r"[A-Za-z]", "", args.backbone.replace("-", "").replace("/", ""))
exp_identifier = f"{args.LLM}SD_{model_version}_{args.dataset}_10"  ##_SD_gen默认目录                        #☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#####################################################################
multi_scale_dir = os.path.join(args.image_root, args.dataset, "muti_scale")
# gen5t_exp_identifier = f"SD_{model_version}_{args.dataset}_5_{gen5t_model_name}_top5"  ##10_gen5t目录
os.makedirs(multi_scale_dir, exist_ok=True)

for scale in range(1, 11):  # 假设跑10轮
    scale_factor = scale / 10
    print(f"Processing scale factor: {scale_factor}")

    # 动态调整 CLIP 特征保存路径 #
    CLIP_feature_gen_path = os.path.join(multi_scale_dir, f"{args.LLM}CLIP_{model_name}_feature_gen{args.Ngen}_scale{scale}.hdf5")                        #☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#####################################################################

    if os.path.exists(CLIP_feature_gen_path):
        print(f" ==> Load existing feature for scale {scale}.")
        hf = h5py.File(CLIP_feature_gen_path, 'r')
        gen_f, gen_l = np.array(hf.get('gen_f')), np.array(hf.get('gen_l'))
        all_embeddings = np.array(hf.get('all_embeddings'))
        print(f" ====> Feature loaded for scale {scale}.")
    else:
        print(f" ==> Extract feature now for scale {scale}.")
        # -------------------- Textual features --------------------#
        print(" ====> Getting textual features from CLIP's text Encoder.")
        all_embeddings = get_textEmbedding(all_names, clip_model, args)
        print("Text embeddings shape: ", all_embeddings.shape)

        # -------------------- Visual features --------------------#
        # Ngen=10时候
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
        # 从JSON文件读取
            json_path = os.path.join(
                args.image_root,
                f"{args.dataset}/json/{args.LLM}SD_{model_version}_{args.dataset}_{model_name}_{args.Ngen}.json"                        #☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#####################################################################
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
    # -------------------- Visual features --------------------#
        # gen_files, gen_labels = [], []
        # for idx, name in enumerate(all_names):
        #     imgDir = f"{args.gen_root_path}/{exp_identifier}/{name}"  ##          导入的图像路径
        #     all_images = sorted(os.listdir(imgDir))  # 确保文件名排序一致
        #     selected_images = all_images[:args.Ngen]  # 提取前 args.Ngen 张图像
        #     for this_img in selected_images:
        #         filePath = os.path.join(imgDir, this_img)
        #         gen_files.append(filePath)
        #     assert len(gen_files) == (idx + 1) * args.Ngen  # 确保提取的图像数量正确
        #     gen_labels += [idx] * args.Ngen
        # assert len(gen_files) == len(all_names) * args.Ngen  # 最终的断言检查
        # gen_labels = np.array(gen_labels)
        # gendf = list(zip(gen_files, gen_labels))

        # 动态调整图像预处理的缩放参数
        multi_scale_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=224, scale=(scale_factor, scale_factor), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
        ])

        print(f" ====> Getting visual features from CLIP's visual Encoder for scale {scale}.")
        gen_f, gen_l = get_visualEmbedding(clip_model, gendf, args.device, transform=multi_scale_transform)
        gen_f = gen_f.reshape(len(all_names), args.Ngen, gen_f.shape[-1])
        gen_l = gen_l.reshape(len(all_names), args.Ngen, 1)
        print(f"Gen embeddings shape for scale {scale}: ", gen_f.shape)

        # 保存特征
        with h5py.File(CLIP_feature_gen_path, "w") as f:
            f.create_dataset('gen_f', data=gen_f, compression="gzip")
            f.create_dataset('gen_l', data=gen_l, compression="gzip")
            f.create_dataset('all_embeddings', data=all_embeddings.cpu(), compression="gzip")
            f.close()
            gen_f = torch.from_numpy(gen_f).float().to(args.device)
            gen_l = torch.from_numpy(gen_l).float().to(args.device)
        print(f" ====> Feature saved for scale {scale}.")
        if scale == 10:
            gen_f = gen_f.view(-1, gen_f.size(-1))
            gen_l = gen_l.view(-1)
            simi_scores = torch.matmul(gen_f, all_embeddings.T)
            predicted_classes = torch.argmax(simi_scores, dim=1)
            correct = torch.sum(predicted_classes == gen_l)
            acc = correct.item() / len(gen_l)
            print("[{}]: scale 10 Acc={:.2f}%".format(args.backbone, acc * 100))



dataset_name = args.dataset  # Change this to match the actual dataset
backbone = args.backbone  # Change this to match the actual backbone
base_feature_dir = f"D:\\PYproject\\clipI\\dataset\\{dataset_name}\\muti_scale"   ##生成集特征目录
# save_real_feature_path = f"D:\\PYproject\\clipI\\dataset\\{dataset_name}\\CLIP_{model_name}_feature_multi_scale.hdf5"   ##测试集最终特征保存路径
save_feature_path = f"D:\\PYproject\\clipI\\dataset\\{dataset_name}\\{args.LLM}CLIP_{model_name}_feature_gen{args.Ngen}_ms.hdf5"   ##生成集最终特征保存路径      #☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#☆#####################################################################
# Combine features
combine_multi_scale_features(base_feature_dir, save_feature_path)
print("Combined multi-scale features and labels saved successfully!")