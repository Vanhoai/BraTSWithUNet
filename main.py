import os

import torch
import torch.nn.functional as F

from losses import CombinedLoss
from metrics import SegmentationMetrics
from models import AttentionUNet, ResUNet, TransUNet, UNetBaseline, UNetPlusPlus
from training import OxfordIIITPetMultiClassesTraining

root_directory = os.getcwd()
device = "mps" if torch.mps.is_available() else "cpu"


def plot_training_history(history: dict[str, list[float]]) -> None:
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 4))

    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Plot Dice Score
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["train_dice"], label="Train Dice")
    plt.plot(epochs, history["val_dice"], label="Val Dice")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Score")
    plt.title("Training and Validation Dice Score")
    plt.legend()

    # Plot IoU
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["train_iou"], label="Train IoU")
    plt.plot(epochs, history["val_iou"], label="Val IoU")
    plt.xlabel("Epochs")
    plt.ylabel("IoU")
    plt.title("Training and Validation IoU")
    plt.legend()

    plt.tight_layout()
    plt.show()


def train_binary_oxford_iiit_pet():
    NUM_CLASSES = 3

    model = UNetBaseline(in_channels=3, num_classes=NUM_CLASSES).to(device)
    trainer = OxfordIIITPetMultiClassesTraining(
        root_path=root_directory,
        device=device,
        model=model,
        batch_size=8,
        num_workers=8,
    )

    epochs = 30
    learning_rate = 1e-4
    trainer.train(epochs=epochs, learning_rate=learning_rate)


def run_res_unet():
    NUM_CLASSES = 2
    EPSILON = 1e-7
    THRESHOLD = 0.5
    BATCH_SIZE = 8
    HEIGHT, WIDTH = 128, 128
    CHANNELS = 3
    DEVICE = torch.device("mps" if torch.mps.is_available() else "cpu")

    torch.manual_seed(0)

    # Metrics
    metrics = SegmentationMetrics(threshold=THRESHOLD, epsilon=EPSILON)

    # Loss function
    # criterion = nn.BCEWithLogitsLoss()
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)

    # num_classes = 1 for binary segmentation with BCEWithLogitsLoss
    # model = ResUNet(in_channels=CHANNELS, num_classes=1).to(DEVICE)
    # model = AttentionUNet(in_channels=CHANNELS, num_classes=1).to(DEVICE)
    model = UNetPlusPlus(in_channels=CHANNELS, num_classes=1).to(DEVICE)

    # 1. Init images and sample masks

    # Shape: [batch_size, channels, height, width] : [8, 3, 128, 128]
    images = torch.randn(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH).to(DEVICE)
    # Shape: [batch_size, height, width] : [8, 128, 128]
    masks = torch.randint(0, 2, (BATCH_SIZE, HEIGHT, WIDTH)).long().to(DEVICE)  # {0, 1}

    assert images.shape == (BATCH_SIZE, CHANNELS, HEIGHT, WIDTH)
    assert masks.shape == (BATCH_SIZE, HEIGHT, WIDTH)

    # 2. Forward pass

    # Shape: [batch_size, 1, height, width] : [8, 1, 128, 128]
    pred = model(images)
    assert pred.shape == (BATCH_SIZE, 1, HEIGHT, WIDTH)

    # 3. Compute loss
    # For binary segmentation, use BCEWithLogitsLoss
    pred = pred.squeeze(1)  # Shape: [batch_size, height, width]
    loss = criterion(pred, masks.float())  # Should be a float value

    # BCEWithLogitsLoss: 0.7238839864730835
    # CombinedLoss: 0.5920742750167847

    # 4. Compute metrics
    pred_probs = F.sigmoid(pred)  # Convert logits to probabilities
    iou = metrics.compute_iou(pred_probs, masks)
    dice = metrics.compute_dice_score(pred_probs, masks)

    print("Loss:", loss.item())
    print("IoU:", iou)
    print("Dice:", dice)


if __name__ == "__main__":
    run_res_unet()
