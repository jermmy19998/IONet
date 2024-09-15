import os
import pdb
import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class Build_Dataset(Dataset):
    """
    Args:
        data_csv (pd.DataFrame): CSV file containing image_id, label, and path to the image file.
        feature_dir (str): Directory containing the feature files.
        patch_size (int): Size of the patches to extract from the image.
        label_dict (dict): Dictionary mapping labels to integers.
    """
    def __init__(self, data_csv: pd.DataFrame, feature_dir:str, patch_size:int, label_dict):
        super().__init__()
        self.data_csv = data_csv
        self.feature_dir = feature_dir
        self.label_dict = label_dict
        self.patch_size = patch_size

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        sample = self.data_csv.iloc[idx]
        file_name = str(sample['image_id'])
        data = torch.load(os.path.join(self.feature_dir, file_name + '.pt'), map_location='cpu')
        # pdb.set_trace()
        if self.patch_size == 16:
            features = data['p_16features']
        elif self.patch_size == 8:
            features = data['p_8features']
        else:
            raise ValueError(f"Unsupported patch size: {self.patch_size}")

        random.shuffle(features)

        label = torch.tensor(self.label_dict[sample['label']])
        return file_name, features, label




class Infer_Dataset(Dataset):
    """
    Infer dataset with vector filter
    Args:
        data_csv (pd.DataFrame): CSV file containing image_id and path to the image file.
        p8_feature_dir (str): Directory containing the p8 feature files.
        p16_feature_dir (str): Directory containing the p16 feature files.
    """
    def __init__(self, data_csv: pd.DataFrame, p8_feature_dir: str, p16_feature_dir: str):
        super().__init__()
        self.data_csv = data_csv
        self.p8_feature_dir = p8_feature_dir
        self.p16_feature_dir = p16_feature_dir

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        sample = self.data_csv.iloc[idx]
        file_name = str(sample['image_id'])

        p8_features = torch.load(os.path.join(self.p8_feature_dir, file_name + '.pt'), map_location='cpu')
        p16_features = torch.load(os.path.join(self.p16_feature_dir, file_name + '.pt'), map_location='cpu')

        p8_features = p8_features['p_8features']
        p16_features = p16_features['p_16features']
        

        p8_sum = torch.sum(p8_features, axis=1)
        p16_sum = torch.sum(p16_features, axis=1)

        p8_top_50_percent_idx = np.argsort(p8_sum)[-int(len(p8_sum) * 0.9):]
        p16_top_50_percent_idx = np.argsort(p16_sum)[-int(len(p16_sum) * 0.9):]

        p8_features = p8_features[p8_top_50_percent_idx]
        p16_features = p16_features[p16_top_50_percent_idx]

        return file_name, p8_features, p16_features