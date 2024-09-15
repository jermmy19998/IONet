#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :	step2__extract_features.py
@Time    :	2024/01/12 19:19:53
@Author  :	SeeingTimes
@Version :	1.0
@Contact :	wacto1998@gmail.com
@License :	MIT License
@Description :	

extracting features with ViT(Patch 8, 16)
ref : https://arxiv.org/abs/2010.11929
'''

import os
import gc
import pandas as pd
import pdb
import argparse
import torch
import torchvision.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing
from torch.cuda.amp import autocast
from torchvision import transforms
from timm.models.vision_transformer import VisionTransformer
from PIL import ImageFile, Image
from tqdm import tqdm, trange
import sys


torch.multiprocessing.set_sharing_strategy('file_system')
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 5_000_000_000
os.environ['VIPS_CONCURRENCY'] = '4'
os.environ['VIPS_DISC_THRESHOLD'] = '15gb'
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None  # Set to None to disable the limit


def parse_args():
    parser = argparse.ArgumentParser(description='Extract patches from whole slide images.')
    parser.add_argument('--in_dim', type=int, help='Input dimension of the model.', default=384)
    parser.add_argument('--num_worker', type=int, help='Number of workers for data loading.', default=0)
    parser.add_argument('--base_dir', type=str, help='Base directory for the project.', required=True)
    parser.add_argument('--patch_dir', type=str, help='Directory for patches.', required=True)
    parser.add_argument('--pretrained_ckpt', type=str, help='Pretrained checkpoint directory.', required=True)
    parser.add_argument('--batch_size', type=int, help='Batch size for data loading.', default=512)

    parser.add_argument('--save_dir_p8', type=str, help='Directory to save the extracted features.', required=True)
    parser.add_argument('--save_dir_p16', type=str, help='Directory to save the extracted features.', required=True)

    parser.add_argument('--device', type=str, help='Device to use for training.', default='cuda:0')
    return parser.parse_args()

args = parse_args()


def check_directory_exists(directory, description):
    if not os.path.isdir(directory):
        print(f"Error: The {description} directory '{directory}' does not exist.")
        sys.exit(1)

def check_file_exists(filepath, description):
    if not os.path.isfile(filepath):
        print(f"Error: The {description} file '{filepath}' does not exist.")
        sys.exit(1)


def check_arguments(args):
    check_directory_exists(args.base_dir, "base directory")
    check_directory_exists(args.patch_dir, "patch directory")
    check_directory_exists(args.pretrained_ckpt, "pretrained checkpoint directory")
    check_directory_exists(args.save_dir_p8, "save directory for p8 features")
    check_directory_exists(args.save_dir_p16, "save directory for p16 features")


check_arguments(args)


class PatchDataset(Dataset):
    def __init__(self, df, transform=None):
        super().__init__()
        self.df = df
        self.wsi_id = df['wsi_id']
        self.patch_id = df['patch_id']
        self.image_path = df['image_path']
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        wsi_id = self.wsi_id[idx]
        patch_id = self.patch_id[idx]
        image_path = self.image_path[idx]
        base_name = os.path.basename(image_path)

        parts = base_name.split('_')
        xy_part = parts[-1].split('-')

        x = int(xy_part[0])
        y = int(xy_part[1].split('.')[0])  

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        return wsi_id, patch_id, image, x, y


os.makedirs(args.patch_dir, exist_ok=True)

vit_p8 = VisionTransformer(patch_size=8, embed_dim=args.in_dim, num_heads=6, num_classes=0)
check_file_exists(f'{args.pretrained_ckpt}/dino_vit_small_patch8_ep200.torch', "ViT p8 checkpoint")
vit_p8.load_state_dict(torch.load(f'{args.pretrained_ckpt}/dino_vit_small_patch8_ep200.torch', map_location='cpu'))
vit_p8 = vit_p8.to(args.device)
vit_p8.eval()

vit_p16 = VisionTransformer(patch_size=16, embed_dim=args.in_dim, num_heads=6, num_classes=0)
check_file_exists(f'{args.pretrained_ckpt}/dino_vit_small_patch16_ep200.torch', "ViT p16 checkpoint")
vit_p16.load_state_dict(torch.load(f'{args.pretrained_ckpt}/dino_vit_small_patch16_ep200.torch', map_location='cpu'))
vit_p16 = vit_p16.to(args.device)
vit_p16.eval()


wsi_ls = []
patch_id_ls = []
image_path_ls = []

for folder_name in os.listdir(args.patch_dir):
    folder_path = os.path.join(args.patch_dir, folder_name)

    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            patch_id = os.path.splitext(file_name)[0]
            wsi_id = folder_name

            image_path = os.path.join(folder_path, file_name)

            wsi_ls.append(wsi_id)
            patch_id_ls.append(patch_id)
            image_path_ls.append(image_path)

data = {'wsi_id': wsi_ls, 'patch_id': patch_id_ls, 'image_path': image_path_ls}
df = pd.DataFrame(data)

transform = transforms.Compose([T.ToTensor(), 
                                T.Resize((224, 224), antialias=True), 
                                T.Normalize(mean=[0.2585, 0.2556, 0.2506], 
                                std=[0.229, 0.224, 0.225])])



unique_wsi_id = df['wsi_id'].unique()
for unique_id in tqdm(unique_wsi_id, desc='Processing wsi_ids'):
    file_path = os.path.join(os.path.join(args.base_dir, args.save_dir_p16), f'{unique_id}.pt')
    if os.path.exists(file_path):
        print(f'{unique_id} already exists in {file_path}, skip')
        continue
    subset_df = df[df['wsi_id'] == unique_id].reset_index(drop=True)
    
    patch_dataset = PatchDataset(subset_df, transform=transform)
    loader_p8 = DataLoader(patch_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker, pin_memory=True, drop_last=False)
    loader_p16 = DataLoader(patch_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker, pin_memory=True, drop_last=False)

    coordinates = []
    patch_ids = []

    all_p8_features = torch.empty((0, args.in_dim)).to(args.device, non_blocking=True)
    all_p16_features = torch.empty((0, args.in_dim)).to(args.device, non_blocking=True)

    with torch.no_grad():
        for (wsi_id_p8, patch_id_p8, image_p8, x_p8, y_p8), (wsi_id_p16, patch_id_p16, image_p16, x_p16, y_p16) in zip(loader_p8, loader_p16):
            if wsi_id_p8 != wsi_id_p16:
                raise ValueError("ID should be the same")
            tiles_p8 = image_p8.to(args.device, non_blocking=True) 
            tiles_p16 = image_p16.to(args.device, non_blocking=True)

            with autocast():
                p8_out = vit_p8(tiles_p8) 
                all_p8_features = torch.cat([all_p8_features, p8_out])

                p16_out = vit_p16(tiles_p16)
                all_p16_features = torch.cat([all_p16_features, p16_out])

                coordinates.extend(zip(x_p8.tolist(), y_p8.tolist()))
                patch_ids.extend(patch_id_p8)
                
    if all_p16_features.size(0) != len(coordinates):
        raise ValueError(f"Mismatch between number of features ({all_p16_features.size(0)}) and coordinates ({len(coordinates)})")
        
    if all_p8_features.size(0) != len(coordinates):
        raise ValueError(f"Mismatch between number of features ({all_p8_features.size(0)}) and coordinates ({len(coordinates)})")

    save_path_p8 = os.path.join(args.base_dir, args.save_dir_p8)
    save_path_p16 = os.path.join(args.base_dir, args.save_dir_p16)

    os.makedirs(save_path_p8, exist_ok=True)
    os.makedirs(save_path_p16, exist_ok=True)

    data_to_save_p8 = {
        'p_8features': all_p8_features,
        'coordinates': coordinates,
        'patch_ids': patch_ids
    }
    data_to_save_p16 = {
        'p_16features': all_p16_features,
        'coordinates': coordinates,
        'patch_ids': patch_ids
    }

    pt_file_name_p8 = os.path.join(save_path_p8, f'{unique_id}.pt')
    pt_file_name_p16 = os.path.join(save_path_p16, f'{unique_id}.pt')

    torch.save(data_to_save_p8, pt_file_name_p8)
    torch.save(data_to_save_p16, pt_file_name_p16)
