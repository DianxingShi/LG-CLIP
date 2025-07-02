import re
import os
import json
import torch
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import scipy.io as sio
from torch.utils.data import Dataset,DataLoader


class CUBDataset():
    def __init__(self, args):
        """
            - args: need
                image_root -> local dataset root path
        """
        # get labels & image_names
        labels_df = pd.read_csv(args.image_root + "/CUB/CUB_200_2011/image_class_labels.txt", sep=' ', header=None)
        self.labels = np.array(labels_df.iloc[:, 1]).astype(np.int64).squeeze() - 1

        files_df = pd.read_csv(args.image_root + "/CUB/CUB_200_2011/images.txt", sep=' ', header=None)
        self.image_files = list(files_df.iloc[:, 1])

        def convert_path(image_files, img_dir):
            new_image_files = []
            for image_file in image_files:
                image_file = os.path.join(img_dir, image_file)
                new_image_files.append(image_file)
            return np.array(new_image_files)
        self.image_files = convert_path(self.image_files, args.image_root + f"/CUB/CUB_200_2011/images")

        # get splits
        split_df = pd.read_csv(args.image_root + "/CUB/CUB_200_2011/train_test_split.txt", sep=' ', header=None)
        self.splits = np.array(split_df.iloc[:, 1]) # 1-> training, 0-> test

        self.train_loc = np.where(self.splits == 1)[0]
        self.test_loc = np.where(self.splits == 0)[0]
        
        # data files
        self.train_files = self.image_files[self.train_loc]
        self.test_files = self.image_files[self.test_loc]

        # get label idxs in mat
        self.train_labels = torch.from_numpy(self.labels[self.train_loc]).long()
        self.test_labels = torch.from_numpy(self.labels[self.test_loc]).long()

        # get class_names
        names_df = pd.read_csv(args.image_root + "/CUB/CUB_200_2011/classes.txt", sep=' ', header=None)
        allclasses_name = names_df.iloc[:, 1]
        self.all_names = [re.sub(r'\d+\.', '', name) for name in allclasses_name]
        self.all_names = change_form(self.all_names)


class FLODataset():
    def __init__(self, args):
        """
            - args: need
                image_root -> local dataset root path
        """
        # get labels & image_names
        # labels_mat = sio.loadmat(args.image_root + "/FLO/Flowers102/imagelabels.mat")
        # self.labels = labels_mat['labels'].astype(np.int64).squeeze()-1
        with open(args.image_root + "/FLO/Flowers102/split_OxfordFlowers.json", 'r') as file:
            split = json.load(file)
        train_df = split["train"]
        # val_df = split["val"]
        test_df = split["test"]

        self.train_files = [train_data[0] for train_data in train_df]
        self.train_labels = [train_data[1] for train_data in train_df]

        self.test_files = [test_data[0] for test_data in test_df]
        self.test_labels = [test_data[1] for test_data in test_df]


        def convert_path(image_files, img_dir):
            new_image_files = []
            for image_file in image_files:
                image_file = os.path.join(img_dir, image_file)
                new_image_files.append(image_file)
            return np.array(new_image_files)
        self.train_files = convert_path(self.train_files, args.image_root + f"/FLO/Flowers102/jpg")
        self.test_files = convert_path(self.test_files, args.image_root + f"/FLO/Flowers102/jpg")

        # get label
        self.train_labels = torch.from_numpy(np.array(self.train_labels)).long()
        self.test_labels = torch.from_numpy(np.array(self.test_labels)).long()

        # get class_names
        with open(args.image_root + "/FLO/Flowers102/cat_to_name.json", 'r') as file:
            name_df = json.load(file)
        allclasses_name = [name_df[str(idx)] for idx in range(1, 103)]
        self.all_names = allclasses_name


class PETDataset():
    def __init__(self, args):
        """
            - args: need
                image_root -> local dataset root path
        """
        # get labels & image_names
        with open(args.image_root + "/PET/OxfordPets/split_OxfordPets.json", 'r') as file:
            split = json.load(file)
        train_df = split["train"]
        # val_df = split["val"]
        test_df = split["test"]

        self.train_files = [train_data[0] for train_data in train_df]
        self.train_labels = [train_data[1] for train_data in train_df]

        self.test_files = [test_data[0] for test_data in test_df]
        self.test_labels = [test_data[1] for test_data in test_df]

        def convert_path(image_files, img_dir):
            new_image_files = []
            for image_file in image_files:
                image_file = os.path.join(img_dir, image_file)
                new_image_files.append(image_file)
            return np.array(new_image_files)
        self.train_files = convert_path(self.train_files, args.image_root + f"/PET/OxfordPets/images")
        self.test_files = convert_path(self.test_files, args.image_root + f"/PET/OxfordPets/images")
        # print(f"Number of test images in PET dataset: {len(self.test_files)}")
        # get label
        self.train_labels = torch.from_numpy(np.array(self.train_labels)).long()
        self.test_labels = torch.from_numpy(np.array(self.test_labels)).long()

        # get class_names
        allclasses_name = sorted(set([train_data[2] for train_data in train_df]))
        self.all_names = change_form(allclasses_name)
    
    


class FOODDataset():
    def __init__(self, args):
        """
            - args: need
                image_root -> local dataset root path
        """
        # get labels & image_names
        with open(args.image_root + "/FOOD/split_Food101.json", 'r') as file:
            split = json.load(file)
        train_df = split["train"]
        # val_df = split["val"]
        test_df = split["test"]

        self.train_files = [train_data[0] for train_data in train_df]
        self.train_labels = [train_data[1] for train_data in train_df]

        self.test_files = [test_data[0] for test_data in test_df]
        self.test_labels = [test_data[1] for test_data in test_df]

        def convert_path(image_files, img_dir):
            new_image_files = []
            for image_file in image_files:
                image_file = os.path.join(img_dir, image_file)
                new_image_files.append(image_file)
            return np.array(new_image_files)
        self.train_files = convert_path(self.train_files, args.image_root + f"/FOOD/images")
        self.test_files = convert_path(self.test_files, args.image_root + f"/FOOD/images")

        # get label
        self.train_labels = torch.from_numpy(np.array(self.train_labels)).long()
        self.test_labels = torch.from_numpy(np.array(self.test_labels)).long()

        # get class_names
        with open(args.image_root + "/FOOD/meta/classes.txt", "r") as f:
            lines = f.readlines()
            allclasses_name = [line.strip() for line in lines]
        self.all_names = change_form(allclasses_name)



class ImageNetDataset:
    def __init__(self, args):
        """
        - args: 需要包含以下参数
            image_root -> 数据集的根路径
        """
        # 图像验证集路径
        self.image_root = os.path.join(args.image_root, "ImageNet/images/ILSVRC2012_img_val/")
        self.image_files = []  # 存储图像文件路径
        self.image_classes = []  # 存储图像的类名

        # 遍历文件夹，收集图像路径及对应类名（父目录名作为类名）
        for root, dirs, files in os.walk(self.image_root):
            class_name = os.path.basename(root)  # 获取父目录名作为类名
            for file in files:
                if file.endswith(".JPEG"):
                    self.image_files.append(os.path.join(root, file))
                    self.image_classes.append(class_name)

        # 按文件名排序，确保与类名一致
        sorted_indices = sorted(range(len(self.image_files)), key=lambda i: self.image_files[i])
        self.image_files = [self.image_files[i] for i in sorted_indices]
        self.image_classes = [self.image_classes[i] for i in sorted_indices]

        # 提取所有唯一的类名，并分配顺序索引
        self.all_names = sorted(set(self.image_classes))  # 排序后去重
        self.class_to_idx = {name: idx for idx, name in enumerate(self.all_names)}  # 类名到索引映射

        # 为每张图像分配标签
        self.train_labels = [self.class_to_idx[class_name] for class_name in self.image_classes]

        # 将验证集作为训练集和测试集
        self.train_files = self.image_files  # 图像路径
        self.train_prompts = self.image_classes  # 类名作为 prompt
        self.train_labels = torch.tensor(self.train_labels).long()

        self.test_files = self.train_files
        self.test_prompts = self.train_prompts
        self.test_labels = self.train_labels


class EUROSATDataset():
    def __init__(self, args):
        """
        - args: 需要包含以下参数
            image_root -> 数据集的根路径
        """
        # 加载数据集划分
        with open(args.image_root + "/EUROSAT/split_EuroSAT.json", 'r') as file:
            split = json.load(file)
        
        # 获取训练集和测试集图像文件及标签
        train_df = split["train"]
        test_df = split["test"]

        self.train_files = [train_data[0] for train_data in train_df]
        self.train_labels = [train_data[1] for train_data in train_df]

        self.test_files = [test_data[0] for test_data in test_df]
        self.test_labels = [test_data[1] for test_data in test_df]

        def convert_path(image_files, img_dir):
            new_image_files = []
            for image_file in image_files:
                image_file = os.path.join(img_dir, image_file)
                new_image_files.append(image_file)
            return np.array(new_image_files)
        self.train_files = convert_path(self.train_files, args.image_root + f"/EUROSAT/2750")
        self.test_files = convert_path(self.test_files, args.image_root + f"/EUROSAT/2750")
        # print(f"Number of test images in PET dataset: {len(self.test_files)}")
        # get label
        self.train_labels = torch.from_numpy(np.array(self.train_labels)).long()
        self.test_labels = torch.from_numpy(np.array(self.test_labels)).long()

        # get class_names
        allclasses_name = sorted(set([train_data[2] for train_data in train_df]))
        self.all_names = change_form(allclasses_name)


class getDataset(Dataset):
    def __init__(self, image_files, labels, transform=None):
        self.image_files = image_files
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        img_pil = Image.open(self.image_files[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img_pil)
        label = self.labels[idx]
        return img, label

    def __len__(self):
        return len(self.image_files)


def change_form(names):
    new_names = []
    for name in names:
        parts = name.split('_')
        name = ' '.join(parts)
        new_names.append(name)
    return new_names

