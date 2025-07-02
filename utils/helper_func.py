import os 
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from scipy.special import softmax


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def compute_accuracy(pred_labels, true_labels, labels):
    acc_per_class = np.zeros(labels.shape[0])
    for i in range(labels.shape[0]):
        idx = (true_labels == labels[i])
        acc_per_class[i] = np.sum(pred_labels[idx] == true_labels[idx]) / np.sum(idx)
    return np.mean(acc_per_class)

def l2_norm2(input):
    norm = torch.norm(input, 2, -1, True)
    output = torch.div(input, norm)
    return output


def create_unique_folder_name(base_folder_path):
    count = 0
    new_folder_name = base_folder_path
    while os.path.exists(new_folder_name):
        count += 1
        new_folder_name = f"{base_folder_path}({count})"
    return new_folder_name

def change_form(names):
    new_names = []
    for name in names:
        parts = name.split('_')
        name = ' '.join(parts)
        new_names.append(name)
        # if len(parts) > 1:
        #     new_name = [parts[0]] + ["-" + parts[i] for i in range(1, len(parts) - 1)] + [" " + parts[-1]]
        #     new_names.append(''.join(new_name))
        # else:
        #     new_names.append(name)
    return new_names


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images