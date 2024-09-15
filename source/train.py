import os
import random
import numpy as np
import pandas as pd
import argparse
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import balanced_accuracy_score
from dataset import Build_Dataset,Infer_Dataset
from model import ABMIL, DSMIL, IClassifier, BClassifier,TransMIL, CLAM_SB,CLAM_MB
from engine import train_one_epoch, valid_one_epoch


def parse_args():
    parser = argparse.ArgumentParser(description='Extract patches from whole slide images.')
    
    parser.add_argument('--base_dir', type=str, help='Base directory for the project.')
    parser.add_argument('--patch_dir', type=str, help='Directory for patches.')
    parser.add_argument('--train_csv', type=str, help='Path to training CSV file')
    parser.add_argument('--test_csv', type=str, help='Path to testing CSV file')
    parser.add_argument('--feature_dir', type=str, help='Directory for features.')
    parser.add_argument('--save_ckpt_dir', type=str, help='Path to save the model.')

    # train parameters
    parser.add_argument('--in_dim', type=int, help='Input dimension of the model.',default=384)
    parser.add_argument('--epochs', type=int, help='Number of epochs to train.',default=20)
    parser.add_argument('--batch_size', type=int, help='Number of batch_size to train.',default=1)
    parser.add_argument('--patch_size', type=int, help='ViT patch size for model to use.',default=16)
    parser.add_argument('--mil_type', type=str, help='Type of MIL model to use.',default='abmil')
    parser.add_argument('--accumulate', type=bool, help='Whether to accumulate gradients.',default=True)
    parser.add_argument('--validation', type=bool, default=False)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--save_best', type=bool, help='Whether to save best weights', default=False)
    parser.add_argument('--DEBUG', type=bool, default=False)


    # hardware
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_worker', type=int, help='Number of workers for data loading.',default=0)

    return parser.parse_args()


def main(args):
    train_csv = pd.read_csv(args.train_csv)
    test_csv = pd.read_csv(args.test_csv)

    label_names = ['HGSC', 'CCOC', 'LGSC','ECOC']
    print(train_csv['label'].value_counts())
    #! debug
    if args.DEBUG:
        train_csv = train_csv.iloc[:1,:]

    label_dict = {label_names[i]: i for i in range(len(label_names))}
    args.feature_dir = os.path.join(args.base_dir, f'sc_40x_ViT_p{args.patch_size}_448_0.5')

    save_path = os.path.join(args.save_ckpt_dir, f'final_wsi_vitp{args.patch_size}_{args.epochs}ep.pth')


    train_dataset = Build_Dataset(data_csv=train_csv, 
                                  feature_dir=args.feature_dir, 
                                  patch_size=args.patch_size,
                                  label_dict=label_dict)

    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              num_workers=args.num_worker, 
                              pin_memory=True)

    val_dataset = Build_Dataset(data_csv=test_csv, 
                                feature_dir=args.feature_dir, 
                                patch_size=args.patch_size,
                                label_dict=label_dict)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_worker, pin_memory=True)

    if args.mil_type == 'abmil':
        model = ABMIL(args.in_dim, 512, 128, args.num_classes)
    elif args.mil_type == 'dsmil':
        model = DSMIL(IClassifier(args.in_dim, args.num_classes), BClassifier(args.in_dim, args.num_classes))
    elif args.mil_type == 'transmil':
        model = TransMIL(args.in_dim, args.num_classes,device=args.device)
    elif args.mil_type == 'clam_sb':
        model = CLAM_SB(size_arg='small',dropout = True,k_sample=8,n_classes=args.num_classes, subtyping=True,embed_dim=args.in_dim)
    elif args.mil_type == 'clam_mb':
        model = CLAM_MB(size_arg='big',dropout = True,k_sample=8,n_classes=args.num_classes, subtyping=True,embed_dim=args.in_dim)


    optimizer = optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.lr)
    model = model.to(args.device)

    
    best_bacc = 0.0
    lowest_loss = float('inf')
    best_model_path = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_bacc = train_one_epoch(model, train_loader, args.device, optimizer, args.mil_type, args.accumulate, epoch, args.epochs)
        valid_loss, valid_bacc = valid_one_epoch(model, val_loader, args.device, args.mil_type, epoch, args.epochs)

        scheduler.step()
        
        if args.save_best:
            if valid_bacc > best_bacc or valid_loss < lowest_loss:
                best_bacc = max(best_bacc, valid_bacc)
                lowest_loss = min(lowest_loss, valid_loss)
                model_name = f"{args.mil_type}{args.patch_size}_epoch_{epoch}_loss_{valid_loss:.4f}_bacc_{valid_bacc:.4f}.pth"
                save_path = os.path.join(args.save_ckpt_dir, model_name)
                os.makedirs(args.save_ckpt_dir,exist_ok=True)
                torch.save(model.state_dict(), save_path)
                best_model_path = save_path
            
                print(f"Best model saved in {best_model_path}")
        else:
            model_name = f"{args.mil_type}{args.patch_size}_epoch_{epoch}_loss_{valid_loss:.4f}_bacc_{valid_bacc:.4f}.pth"
            save_path = os.path.join(args.save_ckpt_dir, model_name)
            os.makedirs(args.save_ckpt_dir,exist_ok=True)
            torch.save(model.state_dict(), save_path)



if __name__ == '__main__':
    args = parse_args()
    main(args)