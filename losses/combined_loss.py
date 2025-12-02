import torch
import torch.nn as nn
from .dice_loss import DiceLoss


class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (batch, num_classes, H, W) - logits
            target: (batch, num_classes, H, W) - one-hot encoded
        """
        # CrossEntropy Loss expects class indices, not one-hot encoded targets
        target_indices = torch.argmax(target, dim=1)  # (batch, H, W)

        ce = self.ce_loss(pred, target_indices)
        dice = self.dice_loss(pred, target)

        return self.ce_weight * ce + self.dice_weight * dice
