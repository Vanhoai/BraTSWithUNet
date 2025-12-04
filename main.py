import os
import torch

from models import UNetBaseline
from training import OxfordIIITPetTraining, OxfordIIITPetMultiClassesTraining

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


if __name__ == "__main__":
    train_binary_oxford_iiit_pet()
