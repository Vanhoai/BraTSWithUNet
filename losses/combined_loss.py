import torch
import torch.nn as nn
from .dice_loss import DiceLoss, MultiClassDiceLoss


class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred, target) -> torch.Tensor:
        bce = self.bce_loss(pred, target.float())
        dice = self.dice_loss(pred, target.float())

        combined_loss = self.bce_weight * bce + self.dice_weight * dice
        return combined_loss


class MultiClassCombinedLoss(nn.Module):
    def __init__(self, num_classes: int, ce_weight=0.5, dice_weight=0.5):
        super(MultiClassCombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = MultiClassDiceLoss(num_classes)

    def forward(self, pred, target):
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)

        combined_loss = self.ce_weight * ce + self.dice_weight * dice
        return combined_loss
