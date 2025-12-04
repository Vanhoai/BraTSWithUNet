import torch


class SegmentationMetrics:
    def __init__(self, num_classes: int = 2, epsilon: float = 1e-7):
        self.num_classes = num_classes
        self.epsilon = epsilon

    def __calculate_binary_iou(self, pred, target) -> torch.Tensor:
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

    def calculate_iou(self, pred, target) -> torch.Tensor:
        """
        Calculate Intersection over Union (IoU) metric.
        For binary: pred and target are (B, H, W) with values {0, 1}
        For multi-class: pred and target are (B, H, W) with class indices
        Returns:
            Mean IoU score across all classes
        """
        if self.num_classes == 2:
            # Binary IoU
            return self.__calculate_binary_iou(pred, target)
        else:
            # Multi-class IoU
            ious = []

            return torch.mean(torch.stack(ious))

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
        dice = (2.0 * intersection + self.epsilon) / (
            pred.sum() + target.sum() + self.epsilon
        )

        return dice
