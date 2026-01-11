"""
This module performs K-fold cross-validation training, validation, and testing
of an image segmentation model using PyTorch. It manages data loading,
training loops, model evaluation, and logging of per-fold metrics.

The training pipeline includes:
    - Reproducible seeding
    - K-fold dataset splitting
    - Model training and validation per epoch
    - Metrics logging and saving for each fold
    - Aggregation of fold-level test results
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold, train_test_split
from losses import DiceLoss, HybridLoss
from dataset import CSVDataset
from model import *
from train import train_one_epoch, validate, test
from config import config_args
from typing import Any


def run_training(args: Any) -> None:
    """
    Execute K-fold cross-validation training and evaluation.

    Args:
        args: Object containing configuration parameters such as dataset directories and image settings.
    Returns:
        None. Results and logs are written to disk.
    """
    # Restrict visible GPUs and set the target device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_name
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Your model is running on {DEVICE}...\n")

    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. Load dataset
    dataset_path = os.path.join(args.dataset_dir, args.csv_file)
    df = pd.read_csv(dataset_path)
    dataset = CSVDataset(args, df)

    # 2. Get K Folds
    training_validation_df, testing_df = train_test_split(df, test_size=0.2, random_state=args.seed)
    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)

    # 3. Version output directory
    version_output_dir = os.path.join(args.output_dir, args.version)
    os.makedirs(version_output_dir, exist_ok=True)

    # 4. K-Fold Cross Validation
    fold_results = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(training_validation_df)):
        print(f"Training fold {fold + 1}/{args.num_folds}...")

        # Split for train and test
        train_indices = training_validation_df.iloc[train_idx].reset_index(drop=True)
        val_indices = training_validation_df.iloc[test_idx].reset_index(drop=True)

        train_ds = CSVDataset(args, train_indices)
        val_ds = CSVDataset(args, val_indices)
        test_ds = CSVDataset(args, testing_df.reset_index(drop=True))

        num_workers = 1

        train_loader = DataLoader(
            train_ds, batch_size=args.batch, shuffle=True, num_workers=num_workers
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch, shuffle=False, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch, shuffle=False, num_workers=num_workers
        )

        # 5. Init model and optimizer and loss function
        model = None
        criterion = DiceLoss()
        if(args.version == "Model_1"):
            model = UNet(num_classes=args.num_classes).to(DEVICE)
        elif(args.version == "Model_2"):
            model = ResUNet1(num_classes=args.num_classes).to(DEVICE)
        elif(args.version == "Model_3"):
            model = ResUNet2(num_classes=args.num_classes).to(DEVICE)
        elif(args.version == "Model_4"):
            model = ResUNet3(num_classes=args.num_classes).to(DEVICE)
        elif(args.version == "Model_5"):
            model = ResUNet4(num_classes=args.num_classes).to(DEVICE)
            criterion = HybridLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        fold_metrics = []

        for epoch in range(args.epochs):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )
            val_loss, val_acc, val_dice, val_iou, val_fpr, val_fnr = validate(
                model, val_loader, criterion, DEVICE
            )

            print(
                f"Fold {fold + 1}, Epoch {epoch + 1}/{args.epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Val DICE: {val_dice:.4f}, Val IoU: {val_iou:.4f}, "
                f"Val FPR: {val_fpr:.4f}, Val FNR: {val_fnr:.4f}"
            )

            fold_metrics.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_dice": val_dice,
                    "val_iou": val_iou,
                    "val_fpr": val_fpr,
                    "val_fnr": val_fnr,
                }
            )
        
        # save fold metrics
        fold_df = pd.DataFrame(fold_metrics)
        metrics_path = os.path.join(version_output_dir, f"fold_{fold + 1}_training_metrics.csv")
        fold_df.to_csv(metrics_path, index=False)

        # Test the model
        test_acc, test_dice, test_iou, test_fpr, test_fnr = test(
            model, test_loader, DEVICE, os.path.join(version_output_dir, f"fold_{fold + 1}_test_outputs")
        )
        fold_results.append(
            {
                "fold": fold + 1,
                "test_acc": test_acc,
                "test_dice": test_dice,
                "test_iou": test_iou,
                "test_fpr": test_fpr,
                "test_fnr": test_fnr,
            }
        )

    # Save fold results
    fold_results_df = pd.DataFrame(fold_results)
    results_path = os.path.join(version_output_dir, f"fold_test_results.csv")
    fold_results_df.to_csv(results_path, index=False)

if __name__ == "__main__":
    args = config_args.parse_args()
    run_training(args)
