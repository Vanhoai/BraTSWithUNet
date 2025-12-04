import torch
import torch.nn.functional as F


class MultiClassSegmentationMetrics:
    def __init__(self, num_classes: int, epsilon: float = 1e-7):
        self.num_classes = num_classes
        self.epsilon = epsilon

    def compute_dice_score(self, pred: torch.Tensor, target) -> float:
        """
        Compute average Dice Score for multi-class segmentation.
        Args:
            pred: Predicted mask (B, C, H, W) - logits (after softmax)
            target: Ground truth mask (B, H, W) - ground truth class indices {0, 1, ..., C-1}
        """
        pred_classes = torch.argmax(pred, dim=1)  # [B, H, W]

        # Convert pred and target to one-hot encoding
        # [B, H, W, C] -> [B, C, H, W]
        pred_oh = F.one_hot(pred_classes, self.num_classes)
        pred_oh = pred_oh.permute(0, 3, 1, 2).float()

        target_oh = F.one_hot(target, self.num_classes)
        target_oh = target_oh.permute(0, 3, 1, 2).float()

        # Compute Dice for each class
        dice_scores = []
        for cls in range(self.num_classes):
            pred_cls = pred_oh[:, cls, :, :].contiguous().view(-1)
            target_cls = target_oh[:, cls, :, :].contiguous().view(-1)

            intersection = (pred_cls * target_cls).sum()
            dice = (2. * intersection + self.epsilon) / (pred_cls.sum() + target_cls.sum() + self.epsilon)
            dice_scores.append(dice)

        # Average Dice score over all classes
        mean_dice = torch.stack(dice_scores).mean()
        return mean_dice.item()

    def compute_iou(self, pred, target) -> float:
        """
        Compute average IoU for multi-class segmentation.
        Args:
            pred: Predicted mask (B, C, H, W) - logits (after softmax)
            target: Ground truth mask (B, H, W) - ground truth class indices {0, 1, ..., C-1}
        """
        pred_classes = torch.argmax(pred, dim=1)  # [B, H, W]

        # Convert pred and target to one-hot encoding
        pred_oh = F.one_hot(pred_classes, self.num_classes)
        pred_oh = pred_oh.permute(0, 3, 1, 2).float()

        target_oh = F.one_hot(target, self.num_classes)
        target_oh = target_oh.permute(0, 3, 1, 2).float()

        # Compute IoU for each class
        iou_scores = []
        for cls in range(self.num_classes):
            pred_cls = pred_oh[:, cls, :, :].contiguous().view(-1)
            target_cls = target_oh[:, cls, :, :].contiguous().view(-1)

            intersection = (pred_cls * target_cls).sum()
            union = pred_cls.sum() + target_cls.sum() - intersection
            iou = (intersection + self.epsilon) / (union + self.epsilon)
            iou_scores.append(iou)

        # Average IoU over all classes
        mean_iou = torch.stack(iou_scores).mean()
        return mean_iou.item()
