import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss cho Segmentation
    Dice = 2 * |A ∩ B| / (|A| + |B|)
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (batch, num_classes, H, W) - logits
            target: (batch, num_classes, H, W) - one-hot encoded
        """
        pred = torch.softmax(pred, dim=1)  # Convert logits to probabilities

        # Flatten
        pred = pred.view(pred.size(0), pred.size(1), -1)  # (batch, classes, H*W)
        target = target.view(target.size(0), target.size(1), -1)

        # Dice score
        intersection = (pred * target).sum(dim=2)
        union = pred.sum(dim=2) + target.sum(dim=2)  # (batch, classes)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()  # Mean across all classes and batch

        return dice_loss
