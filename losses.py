"""
This module calculates DICE Loss and Combo Loss from logits and targets.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W]  (raw outputs from model)
        targets: [B, H, W]    (integer class labels)
        """
        probabilites = F.softmax(logits, dim=1)

        one_hot_targets = F.one_hot(targets, num_classes=2)
        one_hot_targets = one_hot_targets.permute(0, 3, 1, 2).float()

        probabilites = probabilites.view(probabilites.size(0), probabilites.size(1), -1)
        one_hot_targets = one_hot_targets.view(
            one_hot_targets.size(0), one_hot_targets.size(1), -1
        )

        intersection = (probabilites * one_hot_targets).sum(dim=2)
        dice = (2 * intersection + self.smooth) / (
            probabilites.sum(dim=2) + one_hot_targets.sum(dim=2) + self.smooth
        )

        dice_loss = 1 - dice.mean()

        return dice_loss

class HybridLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        """
        Combination of Dice Loss and Cross Entropy Loss.
        """
        super(HybridLoss, self).__init__()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.ce_loss = nn.CrossEntropyLoss()
        self.weight_dice = 1.0
        self.weight_ce = 1.0

    def forward(self, logits, targets):
        """
        logits: [B, C, H, W]  (raw outputs from model)
        targets: [B, H, W]    (integer class labels)
        """
        dice = self.dice_loss(logits, targets)
        ce = self.ce_loss(logits, targets)
        
        combo = self.weight_dice * dice + self.weight_ce * ce
        
        return combo