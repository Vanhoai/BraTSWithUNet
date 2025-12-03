import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int = 2, smooth: float = 1e-6):
        self.num_classes = num_classes
        self.smooth = smooth

    def reset(self):
        self.tp = 0  # True Positive
        self.fp = 0  # False Positive
        self.tn = 0  # True Negative
        self.fn = 0  # False Negative

    @staticmethod
    def __flatten(tensor: torch.Tensor):
        return tensor.view(-1)

    # def confusion_matrix_values(self, pred: torch.Tensor, target: torch.Tensor):
    #     """
    #     Calculate True Positive, False Positive, True Negative, False Negative

    #     Args:
    #         pred: (batch, H, W) - predicted binary mask {0, 1}
    #         target: (batch, H, W) - ground truth binary mask {0, 1}

    #     Returns:
    #         tp, fp, tn, fn
    #     """
    #     pred = self.__flatten(pred)
    #     target = self.__flatten(target)

    #     # True Positive: pred=1 & target=1
    #     tp = ((pred == 1) & (target == 1)).sum().float()

    #     # False Positive: pred=1 & target=0
    #     fp = ((pred == 1) & (target == 0)).sum().float()

    #     # True Negative: pred=0 & target=0
    #     tn = ((pred == 0) & (target == 0)).sum().float()

    #     # False Negative: pred=0 & target=1
    #     fn = ((pred == 0) & (target == 1)).sum().float()

    #     return tp, fp, tn, fn

    # def iou_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
    #     """
    #     Intersection over Union (IoU) / Jaccard Index
    #     IoU = |A ∩ B| / |A ∪ B|

    #     Args:
    #         pred: Predicted mask (B, H, W) - binary values {0, 1}
    #         target: Ground truth mask (B, H, W) - binary values {0, 1}
    #     """
    #     tp, fp, tn, fn = self.confusion_matrix_values(pred, target)

    #     intersection = tp
    #     union = tp + fp + fn

    #     iou = (intersection + self.smooth) / (union + self.smooth)
    #     return iou.item()

    # def dice_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
    #     """
    #     Dice Coefficient / F1-Score

    #     Dice = 2 * TP / (2*TP + FP + FN)
    #          = 2 * Intersection / (|A| + |B|)

    #     Args:
    #         pred: (batch, H, W) - binary mask
    #         target: (batch, H, W) - binary mask

    #     Returns:
    #         Dice score (0-1)
    #     """
    #     tp, fp, tn, fn = self.confusion_matrix_values(pred, target)

    #     dice = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
    #     return dice.item()

    def calculate_iou(self, pred, target) -> torch.Tensor:
        """
        Calculate Intersection over Union (IoU) metric.
        Args:
            pred: Predicted mask (B, H, W) - binary values {0, 1}
            target: Ground truth mask (B, H, W) - binary values {0, 1}
        Returns:
            Mean IoU score across the batch
        """

        # Flatten the tensors
        pred = pred.view(-1)
        target = target.view(-1)

        # Calculate intersection and union
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection

        # Add small epsilon to avoid division by zero
        epsilon = 1e-7
        iou = (intersection + epsilon) / (union + epsilon)

        return iou

    def calculate_dice(self, pred, target) -> torch.Tensor:
        """
        Calculate Dice Coefficient (F1 Score) metric.

        Args:
            pred: Predicted mask (B, H, W) - binary values {0, 1}
            target: Ground truth mask (B, H, W) - binary values {0, 1}

        Returns:
            Mean Dice score across the batch
        """
        # Flatten the tensors
        pred = pred.view(-1)
        target = target.view(-1)

        # Calculate intersection
        intersection = (pred * target).sum()

        # Calculate Dice coefficient
        # Dice = 2 * |A ∩ B| / (|A| + |B|)
        epsilon = 1e-7
        dice = (2.0 * intersection + epsilon) / (pred.sum() + target.sum() + epsilon)

        return dice
