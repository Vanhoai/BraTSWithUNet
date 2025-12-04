import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-7):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, pred, target) -> torch.Tensor:
        """
        Calculate Dice Loss.
        Formula:
            Dice Score = (2 * |A ∩ B|) / (|A| + |B|)
            Dice Loss = 1 - Dice Score

        Args:
            pred: Predicted mask (B, H, W) - logits (before sigmoid)
            target: Ground truth mask (B, H, W) - ground truth binary values {0, 1}
        Returns:
            Dice loss value
        """
        pred = torch.sigmoid(pred)  # Apply sigmoid to get probabilities
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.epsilon) / (pred.sum() + target.sum() + self.epsilon)

        return 1 - dice


class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes: int, epsilon: float = 1e-7):
        super(MultiClassDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, pred, target):
        """
        Compute Dice Loss for multi-class segmentation.
        Formula:
            Dice Score = (2 * |A ∩ B|) / (|A| + |B|)
            Dice Loss = 1 - Dice Score
        Args:
            pred: Predicted mask (B, C, H, W) - logits (before softmax)
            target: Ground truth mask (B, H, W) - ground truth class indices {0, 1, ..., C-1}
        """
        # Convert logits to probabilities
        pred = F.softmax(pred, dim=1)  # [B, C, H, W]

        # Convert target to one-hot encoding
        target_oh = F.one_hot(target, self.num_classes)  # [B, H, W, C]
        target_oh = target_oh.permute(0, 3, 1, 2).float()  # [B, C, H, W]

        # Compute Dice for each class
        dice_scores = []
        for cls in range(self.num_classes):
            pred_cls = pred[:, cls, :, :].contiguous().view(-1)
            target_cls = target_oh[:, cls, :, :].contiguous().view(-1)

            intersection = (pred_cls * target_cls).sum()
            dice = (2. * intersection + self.epsilon) / (pred_cls.sum() + target_cls.sum() + self.epsilon)
            dice_scores.append(dice)

        # Average Dice loss over all classes
        mean_dice = torch.stack(dice_scores).mean()
        return 1 - mean_dice
