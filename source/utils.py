import os
import sys
import torch
import numpy as np
from sklearn.utils import class_weight

def compute_class_weights(labels, num_classes):
    """Compute the class weights
    Args:
        labels (list): List of labels
        num_classes (int): Number of classes
    """
    class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                      classes=np.arange(num_classes),
                                                      y=labels)
    return torch.tensor(class_weights, dtype=torch.float)


def check_file_exists(filepath, description):
    """Helper function to check if a file exists"""
    if not os.path.isfile(filepath):
        print(f"Error: The {description} file '{filepath}' does not exist.")
        sys.exit(1)

def check_directory_exists(directory, description):
    """Helper function to check if a directory exists"""
    if not os.path.isdir(directory):
        print(f"Error: The {description} directory '{directory}' does not exist.")
        sys.exit(1)

def check_arguments(args):
    """
    Function to validate the user input arguments,
    Check if required files and directories exist
    """
    check_directory_exists(args.base_dir, "base directory")
    check_file_exists(args.test_csv, "test CSV")
    check_directory_exists(args.p8_feature_dir, "p8 feature directory")
    check_directory_exists(args.p16_feature_dir, "p16 feature directory")
    
    if args.infer_abmil:
        if args.infer_p8:
            check_file_exists(args.abmil_p8_weight, "ABMIL p8 weight")
        if args.infer_p16:
            check_file_exists(args.abmil_p16_weight, "ABMIL p16 weight")
    
    if args.infer_dsmil:
        if args.infer_p8:
            check_file_exists(args.dsmil_p8_weight, "DSMIL p8 weight")
        if args.infer_p16:
            check_file_exists(args.dsmil_p16_weight, "DSMIL p16 weight")
    
    if not (args.infer_abmil or args.infer_dsmil):
        print("Error: You must specify at least one model type to infer (ABMIL or DSMIL).")
        sys.exit(1)
    
    total_weight = args.abmil_p8 + args.abmil_p16 + args.dsmil_p8 + args.dsmil_p16
    if total_weight == 0:
        print("Error: At least one model weight must be greater than 0.")
        sys.exit(1)
    elif abs(total_weight - 1.0) > 1e-6:
        print("Error: The sum of model weights should be 1.")
        sys.exit(1)