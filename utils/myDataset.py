"""
utils/myDataset.py
------------------
Dataset loader classes for LG-CLIP experiments.

Supported datasets:
  - CUBDataset    : CUB-200-2011 (200 fine-grained bird species)
  - FLODataset    : Oxford 102 Flowers
  - PETDataset    : Oxford-IIIT Pets (37 categories)
  - FOODDataset   : Food-101 (101 food categories)
  - ImageNetDataset: ImageNet ILSVRC2012 validation set
  - EUROSATDataset : EuroSAT (10 satellite image categories)

All loaders expect the following directory structure under `args.image_root`:
  dataset/
    CUB/CUB_200_2011/          # images/, classes.txt, image_class_labels.txt, ...
    FLO/Flowers102/            # jpg/, split_OxfordFlowers.json, cat_to_name.json
    PET/OxfordPets/            # images/, split_OxfordPets.json
    FOOD/                      # images/, split_Food101.json, meta/classes.txt
    ImageNet/images/ILSVRC2012_img_val/  # <synset_id>/*.JPEG
    EUROSAT/                   # 2750/, split_EuroSAT.json
"""

import re
import os
import json
import torch
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader


# ===========================================================================
# CUB-200-2011
# ===========================================================================

class CUBDataset:
    def __init__(self, args):
        """
        Load the CUB-200-2011 dataset.
        Expected directory: <image_root>/CUB/CUB_200_2011/
        Files needed:
          - image_class_labels.txt  (image_id, class_id)
          - images.txt              (image_id, relative_path)
          - train_test_split.txt    (image_id, is_train)
          - classes.txt             (class_id, class_name)
          - images/                 (actual image files)
        """
        base = os.path.join(args.image_root, "CUB", "CUB_200_2011")

        # Labels (1-indexed → 0-indexed)
        labels_df    = pd.read_csv(os.path.join(base, "image_class_labels.txt"), sep=' ', header=None)
        self.labels  = np.array(labels_df.iloc[:, 1]).astype(np.int64).squeeze() - 1

        # Image file paths
        files_df          = pd.read_csv(os.path.join(base, "images.txt"), sep=' ', header=None)
        image_rel_paths   = list(files_df.iloc[:, 1])
        img_root          = os.path.join(base, "images")
        self.image_files  = np.array([os.path.join(img_root, p) for p in image_rel_paths])

        # Train / test split (1 = train, 0 = test)
        split_df     = pd.read_csv(os.path.join(base, "train_test_split.txt"), sep=' ', header=None)
        splits       = np.array(split_df.iloc[:, 1])
        train_loc    = np.where(splits == 1)[0]
        test_loc     = np.where(splits == 0)[0]

        self.train_files  = self.image_files[train_loc]
        self.test_files   = self.image_files[test_loc]
        self.train_labels = torch.from_numpy(self.labels[train_loc]).long()
        self.test_labels  = torch.from_numpy(self.labels[test_loc]).long()

        # Class names (strip numeric prefix, e.g. "001.Black_footed_Albatross" → "Black footed Albatross")
        names_df      = pd.read_csv(os.path.join(base, "classes.txt"), sep=' ', header=None)
        raw_names     = names_df.iloc[:, 1]
        cleaned_names = [re.sub(r'\d+\.', '', name) for name in raw_names]
        self.all_names = change_form(cleaned_names)


# ===========================================================================
# Oxford 102 Flowers
# ===========================================================================

class FLODataset:
    def __init__(self, args):
        """
        Load the Oxford 102 Flowers dataset.
        Expected directory: <image_root>/FLO/Flowers102/
        Files needed:
          - split_OxfordFlowers.json  (train/test splits with [rel_path, label] pairs)
          - cat_to_name.json          (category index → flower name)
          - jpg/                      (image files)
        """
        base = os.path.join(args.image_root, "FLO", "Flowers102")

        with open(os.path.join(base, "split_OxfordFlowers.json"), 'r') as f:
            split = json.load(f)

        img_root = os.path.join(base, "jpg")

        # Extract file paths and labels from split
        train_df   = split["train"]
        test_df    = split["test"]

        self.train_files  = np.array([os.path.join(img_root, d[0]) for d in train_df])
        self.train_labels = torch.from_numpy(np.array([d[1] for d in train_df])).long()

        self.test_files   = np.array([os.path.join(img_root, d[0]) for d in test_df])
        self.test_labels  = torch.from_numpy(np.array([d[1] for d in test_df])).long()

        # Class names (1-indexed JSON keys)
        with open(os.path.join(base, "cat_to_name.json"), 'r') as f:
            name_df = json.load(f)
        self.all_names = [name_df[str(idx)] for idx in range(1, 103)]


# ===========================================================================
# Oxford-IIIT Pets
# ===========================================================================

class PETDataset:
    def __init__(self, args):
        """
        Load the Oxford-IIIT Pets dataset.
        Expected directory: <image_root>/PET/OxfordPets/
        Files needed:
          - split_OxfordPets.json  (train/test splits with [rel_path, label, class_name] triples)
          - images/                (image files)
        """
        base = os.path.join(args.image_root, "PET", "OxfordPets")

        with open(os.path.join(base, "split_OxfordPets.json"), 'r') as f:
            split = json.load(f)

        img_root = os.path.join(base, "images")

        train_df = split["train"]
        test_df  = split["test"]

        self.train_files  = np.array([os.path.join(img_root, d[0]) for d in train_df])
        self.train_labels = torch.from_numpy(np.array([d[1] for d in train_df])).long()

        self.test_files   = np.array([os.path.join(img_root, d[0]) for d in test_df])
        self.test_labels  = torch.from_numpy(np.array([d[1] for d in test_df])).long()

        # Derive class names from the sorted unique set in training split
        self.all_names = change_form(sorted(set([d[2] for d in train_df])))


# ===========================================================================
# Food-101
# ===========================================================================

class FOODDataset:
    def __init__(self, args):
        """
        Load the Food-101 dataset.
        Expected directory: <image_root>/FOOD/
        Files needed:
          - split_Food101.json   (train/test splits)
          - images/              (image files)
          - meta/classes.txt     (101 class names, one per line)
        """
        base = os.path.join(args.image_root, "FOOD")

        with open(os.path.join(base, "split_Food101.json"), 'r') as f:
            split = json.load(f)

        img_root = os.path.join(base, "images")

        train_df = split["train"]
        test_df  = split["test"]

        self.train_files  = np.array([os.path.join(img_root, d[0]) for d in train_df])
        self.train_labels = torch.from_numpy(np.array([d[1] for d in train_df])).long()

        self.test_files   = np.array([os.path.join(img_root, d[0]) for d in test_df])
        self.test_labels  = torch.from_numpy(np.array([d[1] for d in test_df])).long()

        # Class names from text file
        with open(os.path.join(base, "meta", "classes.txt"), "r") as f:
            self.all_names = change_form([line.strip() for line in f.readlines()])


# ===========================================================================
# ImageNet ILSVRC2012 (Validation Set)
# ===========================================================================

class ImageNetDataset:
    def __init__(self, args):
        """
        Load the ImageNet ILSVRC2012 validation set.
        Expected directory: <image_root>/ImageNet/images/ILSVRC2012_img_val/
        Structure: one subfolder per synset, containing .JPEG files.

        Note: the validation set is used for both train and test here
        (zero-shot evaluation scenario, no training needed).
        """
        val_root = os.path.join(
            args.image_root, "ImageNet", "images", "ILSVRC2012_img_val"
        )

        image_files, image_classes = [], []
        for root, dirs, files in os.walk(val_root):
            class_name = os.path.basename(root)   # synset folder name = class label
            for fname in files:
                if fname.endswith(".JPEG"):
                    image_files.append(os.path.join(root, fname))
                    image_classes.append(class_name)

        # Sort by file path for reproducibility
        sorted_idx    = sorted(range(len(image_files)), key=lambda i: image_files[i])
        image_files   = [image_files[i]   for i in sorted_idx]
        image_classes = [image_classes[i] for i in sorted_idx]

        # Build class → index mapping
        self.all_names    = sorted(set(image_classes))
        class_to_idx      = {name: idx for idx, name in enumerate(self.all_names)}

        labels = [class_to_idx[c] for c in image_classes]

        # Use validation set as both train and test (zero-shot evaluation only)
        self.train_files  = image_files
        self.test_files   = image_files
        self.train_labels = torch.tensor(labels).long()
        self.test_labels  = self.train_labels


# ===========================================================================
# EuroSAT
# ===========================================================================

class EUROSATDataset:
    def __init__(self, args):
        """
        Load the EuroSAT dataset.
        Expected directory: <image_root>/EUROSAT/
        Files needed:
          - split_EuroSAT.json  (train/test splits with [rel_path, label, class_name] triples)
          - 2750/               (image files)
        """
        base = os.path.join(args.image_root, "EUROSAT")

        with open(os.path.join(base, "split_EuroSAT.json"), 'r') as f:
            split = json.load(f)

        img_root = os.path.join(base, "2750")

        train_df = split["train"]
        test_df  = split["test"]

        self.train_files  = np.array([os.path.join(img_root, d[0]) for d in train_df])
        self.train_labels = torch.from_numpy(np.array([d[1] for d in train_df])).long()

        self.test_files   = np.array([os.path.join(img_root, d[0]) for d in test_df])
        self.test_labels  = torch.from_numpy(np.array([d[1] for d in test_df])).long()

        # Derive class names from sorted unique set in training split
        self.all_names = change_form(sorted(set([d[2] for d in train_df])))


# ===========================================================================
# Generic Dataset Wrapper (for DataLoader compatibility)
# ===========================================================================

class getDataset(Dataset):
    """A simple wrapper that applies a transform and returns (image, label) pairs."""
    def __init__(self, image_files, labels, transform=None):
        self.image_files = image_files
        self.labels      = labels
        self.transform   = transform

    def __getitem__(self, idx):
        img = Image.open(self.image_files[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.labels[idx]

    def __len__(self):
        return len(self.image_files)


# ===========================================================================
# Utility
# ===========================================================================

def change_form(names: list) -> list:
    """
    Normalize class names: replace underscores with spaces.
    E.g., "Black_footed_Albatross" → "Black footed Albatross"
    """
    return [' '.join(name.split('_')) for name in names]
