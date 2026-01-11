"""
This module calculates relevant metrics from ground truth and predicted labels.
"""

from typing import Sequence, Tuple
import torch
import torch.nn as nn
import numpy as np


def calculate_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Tuple[float, float, float, float, float]:
    """
    This function computes the following metrics based on ground truth labels and predicted labels:
      - Accuracy
      - DICE
      - IoU
      - FPR
      - FNR

    Args:
        y_true (Sequence[int]): Ground Truth binary labels.
        y_pred (Sequence[int]): Predicted binary labels.

    Returns:
        A tuple containing:
        (accuracy, dice, iou, fpr, fnr)
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred)

    dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    fpr = fp / (fp + tn + 1e-6)
    fnr = fn / (fn + tp + 1e-6)
    acc = accuracy_score(y_true, y_pred)

    return acc, dice, iou, fpr, fnr


def confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> Tuple[int, int, int, int]:
    """
    This function computes the following metrics based on ground truth labels and predicted labels:
      - TN
      - FP
      - FN
      - TP

    Args:
        y_true (Sequence[int]): Ground Truth binary labels.
        y_pred (Sequence[int]): Predicted binary labels.

    Returns:
        A tuple containing:
        (tn, fp, fn, tp)
    """
    tn = ((y_true == 0) & (y_pred == 0)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()

    return tn, fp, fn, tp


def accuracy_score(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> float:
    """
    This function computes the accuracy score based on ground truth labels and predicted labels.

    Args:
        y_true (Sequence[int]): Ground Truth binary labels.
        y_pred (Sequence[int]): Predicted binary labels.

    Returns:
        A float value of the accuracy score
    """
    acc = float((y_true == y_pred).sum().item()) / len(y_true)

    return acc
