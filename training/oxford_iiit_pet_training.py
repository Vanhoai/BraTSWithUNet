import os

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import OxfordIIIPetDataset
from metrics import SegmentationMetrics

BATCH_SIZE = 16
NUM_WORKERS = 8


class OxfordIIITPetTraining:
    def __init__(
        self,
        root,
        device,
        model: nn.Module,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
    ):
        self.device = device
        self.model_path = os.path.join(root, "checkpoints", "oxford_iiit_pet")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.model = model.to(self.device)
        transform = A.Compose(
            [
                A.Resize(width=128, height=128),
                A.HorizontalFlip(),
                A.RandomBrightnessContrast(),
                A.Blur(),
                A.Sharpen(),
                A.RGBShift(),
                ToTensorV2(),
            ]
        )

        root_dataset = root + "/data/OxfordIIITPet/oxford-iiit-pet"
        train_dataset = OxfordIIIPetDataset(
            root=root_dataset,
            is_train=True,
            transform=transform,
        )

        val_dataset = OxfordIIIPetDataset(
            root=root_dataset,
            is_train=False,
            transform=transform,
        )

        self.train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=True,
            drop_last=True,
        )

        self.val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=False,
            drop_last=True,
        )

    def train(
        self,
        epochs: int = 50,
        learning_rate: float = 0.0001,
    ):
        # Initialize optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()
        metrics = SegmentationMetrics()

        # Best validation IoU for saving the best model
        best_predict = -1

        for epoch in range(epochs):
            self.model.train()
            train_progress = tqdm(self.train_dataloader, colour="green")

            for batch_idx, (images, masks) in enumerate(train_progress):
                # Move data to device
                images = images.float().to(self.device)  # B, 3, H, W
                masks = masks.float().to(self.device)  # B, H, W

                # Forward pass
                pred = self.model(images)  # B, 1, H, W
                pred = pred.squeeze()  # B, H, W

                # Calculate Loss
                loss = criterion(pred, masks)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                msg = "TRAIN| Epoch: {}/{}| Loss: {:0.4f}".format(epoch, epochs, loss)
                train_progress.set_description(msg)

            # Validation Phase
            self.model.eval()
            all_losses = []
            all_ious = []
            all_dices = []

            val_progress = tqdm(self.val_dataloader, colour="green")

            with torch.no_grad():
                for batch_idx, (images, masks) in enumerate(val_progress):
                    images = images.float().to(self.device)  # B, 3, H, W
                    masks = masks.float().to(self.device)  # B, H, W

                    pred = self.model(images)  # B, 1, H, W
                    pred = pred.squeeze()  # B, H, W

                    loss = criterion(pred, masks)
                    all_losses.append(loss.item())

                    masks = masks.long().cpu()

                    # Make predictions binary
                    pred[pred > 0] = 1  # B, H, W
                    pred[pred < 0] = 0  # B, H, W
                    pred = pred.long().cpu()

                    iou = metrics.calculate_iou(pred, masks)
                    dice = metrics.calculate_dice(pred, masks)

                    all_losses.append(loss.cpu().item())
                    all_ious.append(iou.cpu().item())
                    all_dices.append(dice.cpu().item())

            # Compute mean IoU for the epoch
            loss = np.mean(all_losses)
            miou = np.mean(all_ious)
            dice = np.mean(all_dices)

            msg = "VAL| Loss: {:0.4f} | mIOU: {:0.4f} | Dice: {:0.4f}".format(
                loss,
                miou,
                dice,
            )
            print(msg)

            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "miou": miou,
            }

            # Save Last Checkpoint
            torch.save(checkpoint, os.path.join(self.model_path, "last.h5"))

            # Save best checkpoint based on IoU
            if miou > best_predict:
                torch.save(checkpoint, os.path.join(self.model_path, "best.pth"))
                best_predict = miou
