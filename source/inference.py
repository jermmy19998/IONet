import os
import random
import numpy as np
import pandas as pd
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast
from dataset import Infer_Dataset
from model import ABMIL, DSMIL, IClassifier, BClassifier
from utils import check_arguments
from colorama import Fore, Style, init

def parse_args():
    parser = argparse.ArgumentParser(description='Inference PT files, You should prepare them in advance')
    
    # Directory and file paths
    parser.add_argument('--base_dir', type=str, required=True, help='Base directory for the project.')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to testing CSV file')
    parser.add_argument('--p8_feature_dir', type=str, required=True, help='Directory for p8 features.')
    parser.add_argument('--p16_feature_dir', type=str, required=True, help='Directory for p16 features.')
    parser.add_argument('--save_result_df', type=str, required=True, help='Path for saving the result dataframe')

    # Training parameters
    parser.add_argument('--in_dim', type=int, default=384, help='Input dimension of the model.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument('--num_classes', type=int, default=4)

    # Model weight parameters
    parser.add_argument('--abmil_p8', type=float, default=0., help='Model weight for ABMIL p8')
    parser.add_argument('--abmil_p16', type=float, default=0., help='Model weight for ABMIL p16')
    parser.add_argument('--dsmil_p8', type=float, default=0., help='Model weight for DSMIL p8')
    parser.add_argument('--dsmil_p16', type=float, default=0., help='Model weight for DSMIL p16')

    # Model setup parameters
    parser.add_argument('--infer_p8', default=False, action='store_true', help='Whether to infer p8 model')
    parser.add_argument('--infer_p16', default=False, action='store_true', help='Whether to infer p16 model')
    parser.add_argument('--infer_abmil', default=False, action='store_true', help='Whether to infer ABMIL model')
    parser.add_argument('--infer_dsmil', default=False, action='store_true', help='Whether to infer DSMIL model')

    # Model checkpoint paths
    parser.add_argument('--abmil_p8_weight', type=str, help='Weight path for ABMIL p8')
    parser.add_argument('--abmil_p16_weight', type=str, help='Weight path for ABMIL p16')
    parser.add_argument('--dsmil_p8_weight', type=str, help='Weight path for DSMIL p8')
    parser.add_argument('--dsmil_p16_weight', type=str, help='Weight path for DSMIL p16')

    # Hardware parameters
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_worker', type=int, default=0, help='Number of workers for data loading.')

    return parser.parse_args()

def load_model(model_class, weight_path, device, *args):
    model = model_class(*args)
    model.load_state_dict(torch.load(weight_path, map_location='cpu'))
    model = model.to(device)
    model.eval()
    return model

def build_dataloader(test_csv: pd.DataFrame, p8_feature_dir: str, p16_feature_dir: str, batch_size: int, num_worker: int):
    test_dataset = Infer_Dataset(test_csv, p8_feature_dir, p16_feature_dir)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_worker, pin_memory=True)

def main(args):
    check_arguments(args)

    # Print arguments
    args_dict = vars(args)
    df = pd.DataFrame(list(args_dict.items()), columns=['Argument', 'Value'])
    print(Fore.YELLOW + "# " * 40, Style.RESET_ALL)
    print(Fore.YELLOW + "[CONFIGURATION] Used Arguments", Style.RESET_ALL)
    print(df.to_string(index=False))
    print(Fore.YELLOW + "# " * 40, Style.RESET_ALL)

    # Model information
    used_models = []
    if args.infer_abmil:
        if args.infer_p8:
            used_models.append("ABMIL p8")
        if args.infer_p16:
            used_models.append("ABMIL p16")
    if args.infer_dsmil:
        if args.infer_p8:
            used_models.append("DSMIL p8")
        if args.infer_p16:
            used_models.append("DSMIL p16")

    print(Fore.YELLOW + "[MODEL_INFO] Models to be used: " + ', '.join(used_models), Style.RESET_ALL)
    print(Fore.YELLOW + "# " * 40, Style.RESET_ALL)

    # Load data
    test_csv = pd.read_csv(args.test_csv)
    label_names = ['HGSC', 'CCOC', 'LGSC', 'ECOC']
    result_df = pd.DataFrame(columns=['image_id', 'pred_label', 'HGSC_prob', 'CCOC_prob', 'LGSC_prob', 'ECOC_prob'])
    test_loader = build_dataloader(test_csv, args.p8_feature_dir, args.p16_feature_dir, args.batch_size, args.num_worker)

    # Inference
    with torch.no_grad():
        for ids, p8_features, p16_features in tqdm(test_loader, desc="Evaluating"):
            p8_features = p8_features.squeeze(0).to(args.device)
            p16_features = p16_features.squeeze(0).to(args.device)
            
            # Initialize scores
            abmil_p8_score = torch.zeros((1, args.num_classes)).to(args.device)
            abmil_p16_score = torch.zeros((1, args.num_classes)).to(args.device)
            dsmil_p8_score = torch.zeros((1, args.num_classes)).to(args.device)
            dsmil_p16_score = torch.zeros((1, args.num_classes)).to(args.device)

            if args.infer_abmil:
                if args.infer_p8:
                    abmil_p8 = load_model(ABMIL, args.abmil_p8_weight, args.device, args.in_dim, 512, 128, args.num_classes)
                    abmil_p8_score = torch.softmax(abmil_p8(p8_features), 1)

                if args.infer_p16:
                    abmil_p16 = load_model(ABMIL, args.abmil_p16_weight, args.device, args.in_dim, 512, 128, args.num_classes)
                    abmil_p16_score = torch.softmax(abmil_p16(p16_features), 1)

            if args.infer_dsmil:
                if args.infer_p8:
                    dsmil_p8 = load_model(DSMIL, args.dsmil_p8_weight, args.device, IClassifier(args.in_dim, args.num_classes), BClassifier(args.in_dim, args.num_classes))
                    classes, bag_prediction, _, _ = dsmil_p8(p8_features)
                    max_prediction, _ = torch.max(classes, 0, True)
                    dsmil_p8_score = 0.5 * torch.softmax(max_prediction, 1) + 0.5 * torch.softmax(bag_prediction, 1)

                if args.infer_p16:
                    dsmil_p16 = load_model(DSMIL, args.dsmil_p16_weight, args.device, IClassifier(args.in_dim, args.num_classes), BClassifier(args.in_dim, args.num_classes))
                    classes, bag_prediction, _, _ = dsmil_p16(p16_features)
                    max_prediction, _ = torch.max(classes, 0, True)
                    dsmil_p16_score = 0.5 * torch.softmax(max_prediction, 1) + 0.5 * torch.softmax(bag_prediction, 1)

            # Combine scores and make predictions
            with autocast():
                combined_score = (
                    abmil_p8_score * args.abmil_p8 +
                    abmil_p16_score * args.abmil_p16 +
                    dsmil_p8_score * args.dsmil_p8 +
                    dsmil_p16_score * args.dsmil_p16
                )

                pred = torch.max(combined_score, 1)
                pred_label = label_names[pred.indices]
                score_np = combined_score.cpu().numpy()[0]

                result_df.loc[len(result_df)] = [ids[0], pred_label, score_np[0], score_np[1], score_np[2], score_np[3]]

    # Save results
    result_df.to_csv(args.save_result_df, index=False)
    print(result_df)
    print(Fore.BLUE + "# " * 40, Style.RESET_ALL)
    print(Fore.BLUE + "                                   Completed!", Style.RESET_ALL)
    print(Fore.BLUE + "# " * 40, Style.RESET_ALL)

if __name__ == "__main__":
    args = parse_args()
    main(args)
