import os
import h5py
import numpy as np
import torch
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
    

# Configuration
dataset_name = "FLO"  # Change this to match the actual dataset
backbone = "RN50"  # Change this to match the actual backbone
model_name = backbone.replace("-", "").replace("/", "")
gen5t_model_name = re.sub(r"[A-Za-z]", "", backbone.replace("-", "").replace("/", ""))
base_real_feature_dir = f"D:\\PYproject\\clipI\\dataset\\{dataset_name}\\multi_scale"  ##测试集特征目录
save_real_feature_path = f"D:\\PYproject\\clipI\\dataset\\{dataset_name}\\CLIP_{model_name}_feature_ms.hdf5"   ##测试集最终特征保存路径
# Combine features
combine_multi_scale_features(base_real_feature_dir, save_real_feature_path)
print("Combined multi-scale features and labels saved successfully!")
