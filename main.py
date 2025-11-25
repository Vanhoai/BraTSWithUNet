def dice_loss(pred, target, eps=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)

    intersection = (pred * target).sum()
    dice = (2.0 * intersection + eps) / (pred.sum() + target.sum() + eps)

    return 1 - dice


def multiclass_dice_loss(pred, target, eps=1e-6):
    # pred softmax: [B, C, H, W]
    # target one-hot: [B, C, H, W]
    dice = 0
    num_classes = pred.shape[1]

    for c in range(num_classes):
        p = pred[:, c].contiguous().view(-1)
        t = target[:, c].contiguous().view(-1)

        intersection = (p * t).sum()
        dice_c = (2 * intersection + eps) / (p.sum() + t.sum() + eps)
        dice += 1 - dice_c

    return dice / num_classes


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    # Example usage
    pred = torch.randn(4, 3, 256, 256)  # shape: [B, C, H, W]
    target = torch.randint(0, 3, (4, 256, 256))  # shape: [B, H, W]
    target_one_hot = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()

    pred_softmax = F.softmax(pred, dim=1)
    loss = multiclass_dice_loss(pred_softmax, target_one_hot)
    print(f"Multiclass Dice Loss: {loss.item()}")
