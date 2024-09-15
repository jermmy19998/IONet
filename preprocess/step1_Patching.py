#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    : step1_patch.py
@Time    : 2024/01/12 19:17:03
@Author  : SeeingTimes
@Version : 1.0
@Contact : wacto1998@gmail.com
@License : MIT License
@Description : Extracting patches with pyvips
'''

import os
import time
import random
import pandas as pd
import pyvips
import numpy as np
from PIL import Image
import argparse
from joblib import Parallel, delayed
from tqdm.auto import tqdm

os.environ['VIPS_DISC_THRESHOLD'] = '15gb'

def parse_args():
    parser = argparse.ArgumentParser(description='Extract patches from whole slide images.')
    parser.add_argument('--input_csv', type=str, help='Path to the input CSV file.')
    parser.add_argument('--output_folder', type=str, help='Output folder to save the extracted tiles.')
    parser.add_argument('--tiles_size', type=int, default=448, help='Size for WSI tiles.')
    parser.add_argument('--scale', type=float, default=0.5, help='Scaling factor for resizing tiles.')
    parser.add_argument('--drop_thr', type=float, default=0.7, help='Threshold for dropping tiles based on background in WSI.')
    parser.add_argument('--white_drop_thr', type=float, default=0.7, help='Threshold for dropping tiles based on white background in TMA.')

    return parser.parse_args()

def extract_image_tiles(p_img, folder, wsi_size=768, tma_size=1024, scale=1.0, 
                        drop_thr=0.7, white_drop_thr=0.7, white_thr=210, 
                        wsi_thr=5000, max_samples=50000000000, remove_gray_black=True) -> list:
    """
    Extract image tiles from the input image and save them to the specified folder.

    Args:
        p_img (str): Path to the input image.
        folder (str): Folder to save the extracted tiles.
        wsi_size (int): Size for WSI tiles.
        tma_size (int): Size for TMA tiles.
        scale (float): Scaling factor for resizing tiles.
        drop_thr (float): Threshold for dropping tiles based on background in WSI.
        white_drop_thr (float): Threshold for dropping tiles based on white background in TMA.
        white_thr (int): Threshold for considering a pixel as white in TMA.
        wsi_thr (int): Threshold for distinguishing between WSI and TMA.
        max_samples (int): Maximum number of tiles to extract.
        remove_gray_black (bool): Flag to remove tiles with gray or black backgrounds.

    Returns:
        list: List of paths to the saved tile images.
    """
    name, _ = os.path.splitext(os.path.basename(p_img))
    im = pyvips.Image.new_from_file(p_img)

    if im.width < wsi_thr and im.height < wsi_thr:
        size = tma_size
        scale = 0.5  
    else:
        size = wsi_size

    w = h = size
    idxs = [(y, y + h, x, x + w) for y in range(0, im.height, h) for x in range(0, im.width, w)]
    
    if max_samples < len(idxs):
        idxs = random.sample(idxs, max_samples)
    files = []

    for k, (y, y_, x, x_) in enumerate(idxs):
        tile = im.crop(x, y, min(w, im.width - x), min(h, im.height - y)).numpy()[..., :3]

        if tile.shape[:2] != (h, w):
            continue

        if remove_gray_black:
            gray_black_bg = np.all(tile <= 30, axis=-1)
            if np.sum(gray_black_bg) >= (np.prod(gray_black_bg.shape) * drop_thr):
                continue

        if size == tma_size:
            white_bg = np.all(tile >= white_thr, axis=2)
            if np.sum(white_bg) >= (np.prod(white_bg.shape) * white_drop_thr):
                continue

        if size == wsi_size:
            gray_background = np.all(np.abs(tile - [207, 207, 207]) < 10, axis=-1)
            white_background = np.all(np.abs(tile - [220, 220, 220]) < 30, axis=-1)
            if (np.sum(gray_background) >= (np.prod(gray_background.shape) * drop_thr) or 
                np.sum(white_background) >= (np.prod(white_background.shape) * drop_thr)):
                continue

        new_size = int(size * scale), int(size * scale)
        tile = Image.fromarray(tile).resize(new_size, Image.Resampling.LANCZOS)
        p_img_path = os.path.join(folder, f"{name}_{k:06}_{int(x_ / w)}-{int(y_ / h)}.png") 
        tile.save(p_img_path)
        files.append(p_img_path)

    return files, idxs

def subsample_rand_prune(files: list, max_samples: float = 1.0):
    max_samples = max_samples if isinstance(max_samples, int) else int(len(files) * max_samples)
    random.shuffle(files)
    for p_img in files[max_samples:]:
        os.remove(p_img)

def extract_prune_tiles(
    p_img, folder, size: int = 1024, scale: float = 1.0,
    drop_thr: float = 0.5, max_samples: float = 1.0
) -> None:
    print(f"processing: {p_img}")
    os.makedirs(folder, exist_ok=True)
    tiles, _ = extract_image_tiles(p_img, folder, wsi_size=size, scale=scale, drop_thr=drop_thr)
    subsample_rand_prune(tiles, max_samples)

def main():
    args = parse_args()
    ncoc_csv = pd.read_csv(args.input_csv)
    ls = ncoc_csv['image_path'].tolist()

    def check_file_exists(file_path):
        return os.path.isfile(file_path)

    for index, row in ncoc_csv.iterrows():
        image_path = row['image_path']  
        if not check_file_exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        else:
            print("image path check PASS")

    img_name = lambda p_img: os.path.splitext(os.path.basename(p_img))[0]
    
    _ = Parallel(n_jobs=-1)(
        delayed(extract_prune_tiles)
        (p_img, os.path.join(args.output_folder, img_name(p_img)), size=args.tiles_size, drop_thr=args.drop_thr, scale=args.scale)
        for p_img in tqdm(ls)
    )

    # final patch size would be tiles_size * scale


if __name__ == "__main__":
    """
    input csv should be as follow:

    image_id label	 image_path
    ID_1	 CCOC	/path/to/your/svsfile/ID_1.svs
    ID_2	 CCOC	/path/to/your/svsfile/ID_2.svs
    ID_3	 CCOC	/path/to/your/svsfile/ID_3.svs

    """
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    print(f"Time consumed: {round(total_time / 60, 2)} minutes")
