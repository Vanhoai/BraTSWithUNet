import os

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses import CombinedLoss
from models import UNetBaseline

TRAIN_DATASET_PATH = "./data/Brats2020_TrainingData/MICCAI_BraTS2020_TrainingData"


def pathListIntoIds(dirList):
    x = []
    for i in range(0, len(dirList)):
        x.append(dirList[i][dirList[i].rfind("/") + 1 :])

    return x


def dice_coefficient(pred, target, num_classes=4):
    """
    Calculate Dice Coefficient for each class
    Args:
        pred: (batch, num_classes, H, W) - logits
        target: (batch, num_classes, H, W) - one-hot encoded
        num_classes: number of segmentation classes

    Returns:
        dict: {class_id: dice_score}
    """
    pred = torch.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)  # (batch, H, W)
    target = torch.argmax(target, dim=1)  # (batch, H, W)

    dice_scores = {}
    for class_id in range(num_classes):
        pred_mask = (pred == class_id).float()
        target_mask = (target == class_id).float()

        intersection = (pred_mask * target_mask).sum()
        union = pred_mask.sum() + target_mask.sum()

        if union == 0:
            dice = 1.0  # Perfect score if both are empty
        else:
            dice = ((2.0 * intersection) / union).item()

        dice_scores[class_id] = dice

    return dice_scores


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    pbar = tqdm(dataloader, desc="Training")
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)

        pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_dice_scores = {0: [], 1: [], 2: [], 3: []}

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Calculate metrics
            running_loss += loss.item()
            dice_scores = dice_coefficient(outputs, targets)

            for class_id, score in dice_scores.items():
                all_dice_scores[class_id].append(score)

            avg_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

    # Calculate average dice scores
    avg_dice = {k: np.mean(v) for k, v in all_dice_scores.items()}

    return running_loss / len(dataloader), avg_dice


def save_history(history, filepath="training_history.npy"):
    np.save(filepath, history)
    print(f"Training history saved to {filepath}")


def train_u_net(
    train_dataset,
    val_dataset,
    num_epochs=10,
    batch_size=4,
    learning_rate=1e-4,
    device="cuda",
    save_dir="checkpoints",
):
    os.makedirs(save_dir, exist_ok=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Initialize model
    model = UNetBaseline(in_channels=2, num_classes=4).to(device)

    # Loss and optimizer
    criterion = CombinedLoss(ce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    # Training history
    history = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_val_loss = float("inf")
    print(f"{'=' * 70}")
    print(f"Training Configuration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"{'=' * 70}\n")

    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 70)

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step(val_loss)

        # Save history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)

        # Print results
        print(f"\nResults:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Dice Scores:")
        for class_id, score in val_dice.items():
            class_name = ["Background", "Necrotic", "Edema", "Enhancing"][class_id]
            print(f"    Class {class_id} ({class_name}): {score:.4f}")

        print(f"    Mean Dice: {np.mean(list(val_dice.values())):.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_dice": val_dice,
                },
                save_path,
            )
            print(f"  → Saved best model to {save_path}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch + 1}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                save_path,
            )
    print(f"\n{'=' * 70}")
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'=' * 70}")

    return model, history


def plot_history(history, filepath="training_history.npy"):
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    # Plot Dice Scores
    val_dice_means = [np.mean(list(dice.values())) for dice in history["val_dice"]]
    plt.subplot(1, 2, 2)
    plt.plot(epochs, val_dice_means, label="Mean Val Dice Score", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Dice Score")
    plt.title("Validation Dice Score")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history_plot.png")
    plt.show()


def train_brats():
    # train_and_val_directories = [
    #     f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()
    # ]

    # train_and_test_ids = pathListIntoIds(train_and_val_directories)
    # train_test_ids, val_ids = train_test_split(train_and_test_ids, test_size=0.2)
    # train_ids, test_ids = train_test_split(train_test_ids, test_size=0.15)

    # train_dataset = BrainTumorDataset(ids=train_ids, data_path=TRAIN_DATASET_PATH)
    # val_dataset = BrainTumorDataset(ids=val_ids, data_path=TRAIN_DATASET_PATH)

    # # Train model
    # device = "mps" if torch.mps.is_available() else "cpu"
    # model, history = train_u_net(
    #     train_dataset=train_dataset,
    #     val_dataset=val_dataset,
    #     num_epochs=20,
    #     batch_size=4,
    #     learning_rate=1e-4,
    #     device=device,
    #     save_dir="checkpoints",
    # )

    # save_history(history, filepath="training_history.npy")
    # print("\n✓ Training completed!")

    plot_history(np.load("training_history.npy", allow_pickle=True).item())
