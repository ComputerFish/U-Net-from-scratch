"""
This module provides utility functions for training, validating, and testing
a PyTorch segmentation model. It includes per-epoch training, validation with
metric computation, and final model evaluation.

Functions:
    train_one_epoch: Train the model for a single epoch.
    validate: Evaluate model performance on the validation set.
    test: Evaluate model performance on the test set and save the prediction along with the input for comparison.
"""

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from metrics import calculate_metrics, accuracy_score
from typing import Tuple
from torchvision.utils import save_image
import os


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Train the model for one epoch on the provided training set.

    Args:
        model (nn.Module): The neural network model to train.
        dataloader (DataLoader): DataLoader providing training batches.
        criterion (nn.Module): Loss function used for optimization.
        optimizer (optim.Optimizer): Optimizer for model parameter updates.
        device (torch.device): Device to perform computations on (e.g., 'cuda' or 'cpu').

    Returns:
        Tuple[float, float]: Average training loss and Accuracy for the epoch.
    """
    # Start training
    model.train()

    # Initialize metrics
    all_training_loss = 0.0
    all_predictions = []
    all_labels = []

    # Loop for n number of batches
    for images, masks, image_ids in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        all_training_loss += loss.item() * images.size(0)
        predictions = outputs.argmax(dim=1)
        all_predictions.append(predictions)
        all_labels.append(masks)

    all_predictions = torch.cat(all_predictions).flatten()
    all_labels = torch.cat(all_labels).flatten()

    # get average accuracy and loss for this epoch
    avg_loss = all_training_loss / len(dataloader.dataset)
    acc = accuracy_score(all_predictions, all_labels)

    return avg_loss, acc


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float, float, float, float]:
    """
    Validate the model on the provided validation set and compute detailed metrics.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): DataLoader providing validation data.
        criterion (nn.Module): Loss function used for evaluation.
        device (torch.device): Device to perform computations on.

    Returns:
        Tuple[float, float, float, float, float, float]:
            Average validation loss, Accuracy, DICE, IoU, FPR and FNR.
    """
    model.eval()

    total_validation_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, masks, image_ids in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_validation_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            all_predictions.append(predictions)
            all_labels.append(masks)

    all_predictions = torch.cat(all_predictions).flatten()
    all_labels = torch.cat(all_labels).flatten()

    avg_loss = total_validation_loss / len(dataloader)
    acc, dice, iou, fpr, fnr = calculate_metrics(all_predictions, all_labels)

    return avg_loss, acc, dice, iou, fpr, fnr


def test(
    model: nn.Module, dataloader: DataLoader, device: torch.device, save_image_dir: str
) -> Tuple[float, float, float, float, float]:
    """
    Test the model on the provided testing set and compute detailed metrics.
    Save the models' prediction with the input and ground truth.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): DataLoader providing validation data.
        device (torch.device): Device to perform computations on.
        save_image_dir (str): directory to save the image, mask and predicted mask
                              outputs/model version/image_files/FOLD_n/image_id/[image/mask/pred]
    Returns:
        Tuple[float, float, float, float, float]:
            Accuracy, DICE, IoU, FPR and FNR.
    """
    model.eval()

    all_predictions = []
    all_labels = []

    os.makedirs(save_image_dir, exist_ok=True)

    with torch.no_grad():
        for images, masks, image_ids in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_predictions.append(predictions)
            all_labels.append(masks)

            for i in range(images.size(0)):
                image_id = str(image_ids[i])
                img_save_path = os.path.join(save_image_dir, f"image_{image_id}.png")
                mask_save_path = os.path.join(save_image_dir, f"mask_{image_id}.png")
                pred_save_path = os.path.join(save_image_dir, f"pred_{image_id}.png")

                save_image(images[i].detach().cpu(), img_save_path)
                save_image(masks[i].float().unsqueeze(0), mask_save_path)
                save_image(predictions[i].float().unsqueeze(0), pred_save_path)

    all_predictions = torch.cat(all_predictions).flatten()
    all_labels = torch.cat(all_labels).flatten()

    acc, dice, iou, fpr, fnr = calculate_metrics(all_predictions, all_labels)

    return acc, dice, iou, fpr, fnr
