import os
import torch
import argparse
import numpy as np
import json
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from PIL import Image
from utils.myDataset import *
import clip


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


def get_textEmbedding(classnames, clip_model, args, norm=True):
    """
        - CLIP text embeddings
        - Note: features are normalized
    """
    with torch.no_grad():
        classnames = [classname.replace('_', ' ') for classname in classnames]
        text_descriptions = [f"A photo of a {classname}." for classname in classnames]
        text_tokens = clip.tokenize(text_descriptions, context_length=77).to(args.device)
        text_features = clip_model.encode_text(text_tokens).float().to(args.device)
        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features


# Set paths and args
projectPath = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser(description="Dynamic Filtering and JSON Generation")
parser.add_argument('--dataset', default='ImageNet', help='Dataset name: CUB')
parser.add_argument('--image_root', default=projectPath + '/dataset', help='Path to image root')
parser.add_argument('--gen_root_path', default=projectPath + '/dataset/SD_gen', help='Generated images path')                      
parser.add_argument('--sd_2_1', default=True, action="store_true", help='Use SD 2.1 model')
parser.add_argument('--Ngen', default=10, type=int, help='Number of images to select')
parser.add_argument('--backbone', default='ViT-B/32', help='CLIP backbone')
parser.add_argument('--LLM', default='LLM_', help='LLM_ or None')
parser.add_argument('--device', default='cuda:0', help='Device: cpu/cuda:x')
args = parser.parse_args()

# Set random seed
torch.manual_seed(2024)
np.random.seed(2024)
cudnn.benchmark = True

# Load CLIP
clip_model, preprocess = clip.load(args.backbone, device=args.device)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

# Load dataset
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
    raise ValueError("Unsupported Dataset!")
all_names = mydataset.all_names

# Text features
text_embeddings = get_textEmbedding(all_names, clip_model, args)

# Image paths and labels
model_version = "2.1" if args.sd_2_1 else "1.4"
if args.LLM=="LLM_":
    grp=args.gen_root_path
else:
    grp=projectPath + '/dataset/SD_gen'
gen_root_dir = os.path.join(grp, f"{args.LLM}SD_{model_version}_{args.dataset}_10")                       
if not os.path.exists(gen_root_dir):
    raise FileNotFoundError(f"Default folder {gen_root_dir} does not exist!")

gen_files, gen_labels = [], []
for idx, name in enumerate(all_names):
    img_dir = os.path.join(gen_root_dir, name)
    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Folder for class {name} does not exist!")
    for this_img in os.listdir(img_dir):
        gen_files.append(os.path.join(img_dir, this_img))
    gen_labels += [idx] * 10  # Default: Ngen=10
assert len(gen_files) == len(all_names) * 10

# Visual features
gen_labels = np.array(gen_labels)
gendf = list(zip(gen_files, gen_labels))
gen_f, gen_l = get_visualEmbedding(clip_model, gendf, args.device, transform=preprocess)
gen_f = torch.from_numpy(gen_f).float().to(args.device)
gen_l = torch.from_numpy(gen_l).long().to(args.device)

# Classifier
simi_scores = torch.matmul(gen_f, text_embeddings.T)
predicted_classes = torch.argmax(simi_scores, dim=1)

# Organize results
result_dict = {i: [] for i in range(len(all_names))}
for idx, (gen_file, gen_label) in enumerate(zip(gen_files, gen_l)):
    label = int(gen_label.item())
    prob = simi_scores[idx, label].item()
    result_dict[label].append({"image": gen_file, "probability": prob})

# Filter top Ngen with priority for correct predictions
filtered_result_dict = {}

for label in result_dict:
    
    images = result_dict[label]

   
    correct_images = [img for img in images if predicted_classes[gen_files.index(img["image"])] == label]
    incorrect_images = [img for img in images if img not in correct_images]

    
    correct_images = sorted(correct_images, key=lambda x: x["probability"], reverse=True)
    incorrect_images = sorted(incorrect_images, key=lambda x: x["probability"], reverse=True)

    
    selected_images = correct_images[:args.Ngen]

   
    if len(selected_images) < args.Ngen:
        needed = args.Ngen - len(selected_images)
        selected_images += incorrect_images[:needed]

    
    filtered_result_dict[label] = selected_images

# Save to JSON
json_output_dir = os.path.join(args.image_root, f"{args.dataset}/json")
os.makedirs(json_output_dir, exist_ok=True)
backbone_name = args.backbone.replace("/", "").replace("-", "")
json_output_path = os.path.join(
    json_output_dir,
    f"{args.LLM}SD_{model_version}_{args.dataset}_{backbone_name}_{args.Ngen}.json"                       
)
json_result = {all_names[label]: filtered_result_dict[label] for label in filtered_result_dict}
with open(json_output_path, 'w', encoding='utf-8') as json_file:
    json.dump(json_result, json_file, indent=4, ensure_ascii=False)


