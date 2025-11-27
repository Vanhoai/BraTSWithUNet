import os

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2  # np.array -> torch.tensor
from torch.utils.data import DataLoader
from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU
from tqdm import tqdm

from datasets import OxfordIIIPetDataset
from models import UNetBaseline

# internal modules
from .training import TrainingTorchModel


class OxfordIIITPetTraining(TrainingTorchModel):
    def __init__(
        self,
        saved_directory: str,
        device: str,
    ):
        super().__init__()
        self.model_path = saved_directory
        self.device = device

        best_model_path = os.path.join(self.model_path, "best.pth")
        if not os.path.exists(best_model_path):
            self.build_model()
        else:
            self.load_model()

        if not self.model:
            raise ValueError("Model is not built or loaded properly.")

    def load_model(self):
        checkpoint = torch.load(
            os.path.join(self.model_path, "best.pth"),
            map_location=self.device,
        )
        self.model = self.build_model()
        self.model.load_state_dict(checkpoint["model_state_dict"])

    def save_model(self):
        print("Saving model to: ", self.model_path)

    def build_model(self):
        self.model = UNetBaseline(in_channels=3, num_classes=1)
        self.model = self.model.to(self.device)
        return self.model

    def evaluate(self):
        print("Evaluating the model")

    def train(self):
        LEARNING_RATE = 0.0001
        BATCH_SIZE = 10
        EPOCHS = 50
        NUM_WORKERS = 4

        train_transform = A.Compose(
            [
                A.Resize(width=224, height=224),
                A.HorizontalFlip(),
                A.RandomBrightnessContrast(),
                A.Blur(),
                A.Sharpen(),
                A.RGBShift(),
                ToTensorV2(),
            ]
        )

        root = os.getcwd() + "/data/OxfordIIITPet/oxford-iiit-pet"
        train_dataset = OxfordIIIPetDataset(
            root=root,
            is_train=True,
            transform=train_transform,
        )

        val_dataset = OxfordIIIPetDataset(
            root=root,
            is_train=False,
            transform=train_transform,
        )

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=True,
            drop_last=True,
        )

        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=False,
            drop_last=True,
        )

        # Initialize optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss()

        # Metrics
        miou_metric = MeanIoU(num_classes=2)
        dice_metric = GeneralizedDiceScore(num_classes=2)

        # Best validation IoU for saving the best model
        best_predict = -1

        # Training loop
        for epoch in range(EPOCHS):
            # Training Phase
            self.model.train()
            train_progress = tqdm(train_dataloader, colour="cyan")

            for idx, img_mask in enumerate(train_progress):
                # B, C, H, W
                img = img_mask[0].float().to(self.device)  # type: ignore
                # B, H, W
                mask = img_mask[1].float().to(self.device)

                y_pred = self.model(img)  # B, 1, H, W
                y_pred = y_pred.squeeze()  # B, H, W
                optimizer.zero_grad()

                # Calculate Loss
                loss = criterion(y_pred, mask)

                # Backpropagation
                loss.backward()
                optimizer.step()
                train_progress.set_description(
                    "TRAIN| Epoch: {}/{}| Loss: {:0.4f}".format(epoch, EPOCHS, loss)
                )

            # Validation Phase
            self.model.eval()

            all_losses = []
            all_ious = []
            all_dices = []

            with torch.no_grad():
                for idx, img_mask in enumerate(val_dataloader):
                    img = img_mask[0].float().to(self.device)  # type: ignore
                    mask = img_mask[1].float().to(self.device)  # B W H

                    y_pred = self.model(img)
                    y_pred = y_pred.squeeze()  # B H W

                    loss = criterion(y_pred, mask)

                    mask = mask.long().cpu()
                    y_pred[y_pred > 0] = 1  # BWH
                    y_pred[y_pred < 0] = 0  # BWH
                    y_pred = y_pred.long().cpu()

                    miou = miou_metric(y_pred, mask)
                    dice = dice_metric(y_pred, mask)

                    all_losses.append(loss.cpu().item())
                    all_ious.append(miou.cpu().item())
                    all_dices.append(dice.cpu().item())

                    if idx == 40:
                        break

            # Compute mean IoU for the epoch
            loss = np.mean(all_losses)
            miou = np.mean(all_ious)
            dice = np.mean(all_dices)

            print(
                "VAL| Loss: {:0.4f} | mIOU: {:0.4f} | Dice: {:0.4f}".format(
                    loss, miou, dice
                )
            )

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
