import os
import random
import pdb
import numpy as np
import pandas as pd
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import balanced_accuracy_score
from dataset import Heatmap_Dataset
from model import ABMIL, DSMIL, IClassifier, BClassifier
from utils import draw_heatmap,to_percentiles, A_fliter
from tqdm import tqdm
from torch.cuda.amp import autocast


def parse_args():
    parser = argparse.ArgumentParser(description='Infering pt file,you should prepare the pt file in advance')
    
    parser.add_argument('--base_dir', type=str, help='Base directory for the project.')
    parser.add_argument('--patch_dir', type=str, help='Directory for patches.')
    parser.add_argument('--test_csv', type=str, help='Path to testing CSV file')
    parser.add_argument('--p8_feature_dir', type=str, help='Directory for p8 features.')
    parser.add_argument('--p16_feature_dir', type=str, help='Directory for p16 features.')

    # train parameters
    parser.add_argument('--in_dim', type=int, help='Input dimension of the model.', default=384)
    parser.add_argument('--batch_size', type=int, help='Number of batch size to train.', default=1)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--remove_label', action='store_true', default=True)
    parser.add_argument('--heatmap_save_dir', type=str, help='Directory to save heatmap.')

    # model weight
    parser.add_argument('--abmil_p8', type=float, default=1.0,help='Model weight')
    parser.add_argument('--abmil_p16', type=float, default=1.0,help='Model weight')
    parser.add_argument('--dsmil_p8', type=float, default=1.0,help='Model weight')
    parser.add_argument('--dsmil_p16', type=float, default=1.0,help='Model weight')

    parser.add_argument('--abmil_p8_weight', type=str,help='Weight path for abmil_p8')
    parser.add_argument('--abmil_p16_weight', type=str, help='Weight path for abmil_p16')
    parser.add_argument('--dsmil_p8_weight', type=str, help='Weight path for dsmil_p8')
    parser.add_argument('--dsmil_p16_weight', type=str, help='Weight path for dsmil_p16')

    # hardware
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--num_worker', type=int, help='Number of workers for data loading.', default=0)

    return parser.parse_args()


def build_dataloader(test_csv: pd.DataFrame, p8_feature_dir: str, p16_feature_dir: str):
    test_dataset = Heatmap_Dataset(test_csv, p8_feature_dir, p16_feature_dir) 
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker, pin_memory=True)
    return test_loader


def main():
    args = parse_args()
    
    test_csv = pd.read_csv(args.test_csv)
    if args.remove_label:
        labels_to_remove = ['Outlier', 'MUC']
        test_csv = test_csv[~test_csv['label'].isin(labels_to_remove)]

    label_names = ['HGSC', 'CCOC', 'LGSC', 'ECOC']
    label_dict = {label_names[i]: i for i in range(args.num_classes)}
    result_df = pd.DataFrame(columns=['image_id','pred_label', 'HGSC_prob', 'CCOC_prob', 'LGSC_prob', 'ECOC_prob'])
    test_loader = build_dataloader(test_csv, args.p8_feature_dir, args.p16_feature_dir)

    with torch.no_grad():
        for ids, p8_features, p16_features in tqdm(test_loader, desc="Evaluating"):
            
            p8_features = p8_features.squeeze(0).to(args.device)
            p16_features = p16_features.squeeze(0).to(args.device)

            abmil_p8 = ABMIL(args.in_dim, 512, 128, args.num_classes)
            abmil_p16 = ABMIL(args.in_dim, 512, 128, args.num_classes)
            dsmil_p8 = DSMIL(IClassifier(args.in_dim, args.num_classes), BClassifier(args.in_dim, args.num_classes))
            dsmil_p16 = DSMIL(IClassifier(args.in_dim, args.num_classes), BClassifier(args.in_dim, args.num_classes))

            abmil_p8.load_state_dict(torch.load(args.abmil_p8_weight, map_location='cpu'))
            abmil_p8 = abmil_p8.to(args.device)
            abmil_p8.eval()

            abmil_p16.load_state_dict(torch.load(args.abmil_p16_weight, map_location='cpu'))
            abmil_p16 = abmil_p16.to(args.device)
            abmil_p16.eval()

            dsmil_p8.load_state_dict(torch.load(args.dsmil_p8_weight, map_location='cpu'))
            dsmil_p8 = dsmil_p8.to(args.device)
            dsmil_p8.eval()

            dsmil_p16.load_state_dict(torch.load(args.dsmil_p16_weight, map_location='cpu'))
            dsmil_p16 = dsmil_p16.to(args.device)
            dsmil_p16.eval()

            with autocast():
                abmil_p8_out, ab_A8 = abmil_p8(p8_features)
                abmil_p16_out, ab_A16 = abmil_p16(p16_features)
                abmil_p8_score = torch.softmax(abmil_p8_out, 1)
                abmil_p16_score = torch.softmax(abmil_p16_out, 1)

                classes, bag_prediction, A_8, B_8 = dsmil_p8(p8_features)
                
                max_prediction, idx = torch.max(classes, 0, True)
                dsmil_p8_score = 0.5 * torch.softmax(max_prediction, 1) + 0.5 * torch.softmax(bag_prediction, 1)
                dsmil_8_class,dsmil_p8_index = torch.max(dsmil_p8_score, dim=1)


                classes, bag_prediction, A_16, B_16 = dsmil_p16(p16_features)
                max_prediction, idx = torch.max(classes, 0, True)
                dsmil_p16_score = 0.5 * torch.softmax(max_prediction, 1) + 0.5 * torch.softmax(bag_prediction, 1)
                dsmil_p16_class,dsmil_p16_index = torch.max(dsmil_p16_score, dim=1)

                # attention metrix
                ab_A8 = to_percentiles(ab_A8.view(-1, 1).cpu()) / 100
                ab_A16 = to_percentiles(ab_A16.view(-1, 1).cpu()) / 100
                ds_A8 = to_percentiles(A_8[:,dsmil_p8_index].cpu()) / 100
                ds_A16 = to_percentiles(A_16[:,dsmil_p16_index].cpu()) / 100

                thr = 0.5
                ab_A8 = A_fliter(ab_A8,thr)
                ab_A16 = A_fliter(ab_A16,thr)
                ds_A8 = A_fliter(ds_A8,thr)
                ds_A16 = A_fliter(ds_A16,thr)


                # pdb.set_trace()
                combined_score = (
                    abmil_p8_score * args.abmil_p8 +
                    abmil_p16_score * args.abmil_p16 +
                    dsmil_p8_score * args.dsmil_p8 +
                    dsmil_p16_score * args.dsmil_p16
                )

                pred = torch.max(combined_score, 1)
                pred_label = label_names[pred.indices]
                score_np = combined_score.cpu().numpy()[0]

                attention_matrix_dic_p8 = {
                    'abmil_p8':ab_A8,
                    'dsmil_p8':ds_A8,
                    'all':ab_A8 + ds_A8,
                }

                attention_matrix_dic_p16 = {
                    'abmil_p16':ab_A16,
                    'dsmil_p16':ds_A16,
                    'all':ab_A16 + ds_A16
                }

                filtered_df = test_csv[test_csv['image_id'].isin(ids)]

                paths = filtered_df['image_path'].values.tolist()
                draw_heatmap(save_dir=args.heatmap_save_dir,slide_id=ids[0],patches_dir=args.patch_dir,attention_matrix_dic=attention_matrix_dic_p8,svs_file=paths)
                draw_heatmap(save_dir=args.heatmap_save_dir,slide_id=ids[0],patches_dir=args.patch_dir,attention_matrix_dic=attention_matrix_dic_p16,svs_file=paths)


if __name__ == "__main__":
    args = parse_args()
    main()