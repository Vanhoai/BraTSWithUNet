import torch


class SegmentationMetrics:
    def __init__(self, threshold=0.5, epsilon: float = 1e-7):
        self.threshold = threshold
        self.epsilon = epsilon

    def compute_dice_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute Dice Score.
        Formula:
            Dice Score = (2 * |A ∩ B|) / (|A| + |B|)

        Args:
            pred: Predicted mask (B, H, W) - logits (after sigmoid)
            target: Ground truth mask (B, H, W) - ground truth binary values {0, 1}

        Returns:
            Dice score value
        """
        pred = (pred > self.threshold).float()  # Binarize predictions

        # Flatten tensors
        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.epsilon) / (pred.sum() + target.sum() + self.epsilon)

        return dice.item()

    def compute_iou(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute Intersection over Union (IoU).
        Formula:
            IoU = |A ∩ B| / |A ∪ B|

        Args:
            pred: Predicted mask (B, H, W) - logits (after sigmoid)
            target: Ground truth mask (B, H, W) - ground truth binary values {0, 1}
        Returns:
            IoU score value
        """
        pred_binary = (pred > self.threshold).float()
        pred = pred_binary.view(-1)
        target = target.view(-1)

        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        iou = (intersection + self.epsilon) / (union + self.epsilon)
        return iou.item()

    def compute_pixel_accuracy(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        pred_binary = (pred > self.threshold).float()
        correct = (pred_binary == target).float().sum()  # type: ignore
        total = target.numel()
        return (correct / total).item()
