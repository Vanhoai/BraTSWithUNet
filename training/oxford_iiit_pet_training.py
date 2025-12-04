import os
from typing import Tuple

import albumentations as A
import torch
from albumentations.pytorch import ToTensorV2
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import OxfordIIIPetBinaryDataset, OxfordIIIPetDataset
from losses import CombinedLoss, MultiClassCombinedLoss, MultiClassDiceLoss, DiceLoss
from metrics import SegmentationMetrics, MultiClassSegmentationMetrics


class OxfordIIITPetTraining:
    def __init__(
        self,
        root_path: str,
        device: str,
        model: nn.Module,
        batch_size: int = 16,
        num_workers: int = 8,
    ):
        self.root_path = root_path
        self.model_path = os.path.join(self.root_path, "checkpoints", "oxford_iiit_pet")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model = model

    def __build_dataloaders(self):
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

        root_dataset = os.path.join(self.root_path, "data", "OxfordIIITPet", "oxford-iiit-pet")
        train_dataset = OxfordIIIPetBinaryDataset(
            root=root_dataset,
            is_train=True,
            transform=transform,
        )

        val_dataset = OxfordIIIPetBinaryDataset(
            root=root_dataset,
            is_train=False,
            transform=transform,
        )

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
        )

        return train_dataloader, val_dataloader

    def __train_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        metrics: SegmentationMetrics,
    ):
        self.model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_iou = 0.0
        num_batches = 0

        train_progress = tqdm(dataloader, colour="blue", desc="Training")
        for batch_idx, (images, masks) in enumerate(train_progress):
            # Convert to device
            images = images.float().to(self.device)
            masks = masks.float().to(self.device)

            # Forward pass
            optimizer.zero_grad()
            pred = self.model(images)  # Shape: [B, 1, H, W]
            pred = pred.squeeze(1)  # Shape: [B, H, W]

            # Compute loss
            loss = criterion(pred, masks.float())

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Compute metrics
            with torch.no_grad():
                pred_probs = torch.sigmoid(pred)
                dice = metrics.compute_dice_score(pred_probs, masks)
                iou = metrics.compute_iou(pred_probs, masks)
                acc = metrics.compute_pixel_accuracy(pred_probs, masks)

            epoch_loss += loss.item()
            epoch_dice += dice
            epoch_iou += iou
            num_batches += 1

            msg = (
                f"Training | Loss: {epoch_loss / num_batches:.4f} | "
                f"Dice: {epoch_dice / num_batches:.4f} | "
                f"IoU: {epoch_iou / num_batches:.4f}"
            )
            train_progress.set_description(msg)

        return {
            "loss": epoch_loss / num_batches,
            "dice": epoch_dice / num_batches,
            "iou": epoch_iou / num_batches,
        }

    def __validate(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        metrics: SegmentationMetrics,
    ):
        self.model.eval()
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_iou = 0.0
        num_batches = 0

        with torch.no_grad():
            val_progress = tqdm(dataloader, colour="blue", desc="Validation")
            for batch_idx, (images, masks) in enumerate(val_progress):
                images = images.float().to(self.device)
                masks = masks.float().to(self.device)

                # Forward pass
                pred_logits = self.model(images)  # [B, 1, H, W]
                pred_logits = pred_logits.squeeze(1)  # [B, H, W]

                # Compute loss
                loss = criterion(pred_logits, masks.float())

                # Compute metrics
                pred_probs = torch.sigmoid(pred_logits)
                dice = metrics.compute_dice_score(pred_probs, masks)
                iou = metrics.compute_iou(pred_probs, masks)
                acc = metrics.compute_pixel_accuracy(pred_probs, masks)

                epoch_loss += loss.item()
                epoch_dice += dice
                epoch_iou += iou
                num_batches += 1

                msg = (
                    f"Validation | Loss: {epoch_loss / num_batches:.4f} | "
                    f"Dice: {epoch_dice / num_batches:.4f} | "
                    f"IoU: {epoch_iou / num_batches:.4f}"
                )

                val_progress.set_description(msg)

        return {
            "loss": epoch_loss / num_batches,
            "dice": epoch_dice / num_batches,
            "iou": epoch_iou / num_batches,
        }

    def train(
        self,
        epochs: int = 50,
        learning_rate: float = 1e-4,
    ):
        # Initialize optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        # criterion = nn.BCEWithLogitsLoss()  # Binary Segmentation Loss
        # criterion = DiceLoss()
        criterion = CombinedLoss()

        metrics = SegmentationMetrics()

        # Build Dataloaders
        train_dataloader, val_dataloader = self.__build_dataloaders()

        # Best validation IoU for saving the best model
        best_iou = -1.0
        history = {
            "train_loss": [],
            "train_dice": [],
            "train_iou": [],
            "val_loss": [],
            "val_dice": [],
            "val_iou": [],
        }

        for epoch in range(epochs):
            # Training Phase
            train_metrics = self.__train_epoch(
                dataloader=train_dataloader,
                criterion=criterion,
                optimizer=optimizer,
                metrics=metrics,
            )

            # Validation Phase
            val_metrics = self.__validate(
                dataloader=val_dataloader,
                criterion=criterion,
                metrics=metrics,
            )

            # Save history
            history["train_loss"].append(train_metrics["loss"])
            history["train_dice"].append(train_metrics["dice"])
            history["train_iou"].append(train_metrics["iou"])

            history["val_loss"].append(val_metrics["loss"])
            history["val_dice"].append(val_metrics["dice"])
            history["val_iou"].append(val_metrics["iou"])

            # Save checkpoint
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": val_metrics["iou"],
            }

            # Save Last Checkpoint
            torch.save(checkpoint, os.path.join(self.model_path, "last.pth"))

            # Save best checkpoint based on IoU
            if val_metrics["iou"] > best_iou:
                torch.save(checkpoint, os.path.join(self.model_path, "best.pth"))
                best_iou = val_metrics["iou"]

        return history


class OxfordIIITPetMultiClassesTraining:
    def __init__(
        self,
        root_path: str,
        device: str,
        model: nn.Module,
        batch_size: int = 16,
        num_workers: int = 8,
    ):
        self.root_path = root_path
        self.model_path = os.path.join(self.root_path, "checkpoints", "oxford_iiit_pet")
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model = model

    def __build_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
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

        root_dataset = os.path.join(self.root_path, "data", "OxfordIIITPet", "oxford-iiit-pet")
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

        train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            drop_last=True,
        )

        val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
        )

        return train_dataloader, val_dataloader

    def __train_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        metrics: MultiClassSegmentationMetrics,
    ):
        self.model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_iou = 0.0
        num_batches = 0

        train_progress = tqdm(dataloader, colour="blue", desc="Training")
        for batch_idx, (images, masks) in enumerate(train_progress):
            # Convert to device
            images = images.float().to(self.device)
            masks = masks.long().to(self.device)

            # Forward pass
            optimizer.zero_grad()
            pred = self.model(images)  # [B, C, H, W]

            # Compute loss
            loss = criterion(pred, masks)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Compute metrics
            with torch.no_grad():
                pred_probs = torch.softmax(pred, dim=1)
                dice = metrics.compute_dice_score(pred_probs, masks)
                iou = metrics.compute_iou(pred_probs, masks)

            epoch_loss += loss.item()
            epoch_dice += dice
            epoch_iou += iou
            num_batches += 1

            msg = (
                f"Training | Loss: {epoch_loss / num_batches:.4f} | "
                f"Dice: {epoch_dice / num_batches:.4f} | "
                f"IoU: {epoch_iou / num_batches:.4f}"
            )
            train_progress.set_description(msg)

        return {
            "loss": epoch_loss / num_batches,
            "dice": epoch_dice / num_batches,
            "iou": epoch_iou / num_batches,
        }

    def __validate(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
        metrics: MultiClassSegmentationMetrics,
    ):
        self.model.eval()
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_iou = 0.0
        num_batches = 0

        with torch.no_grad():
            val_progress = tqdm(dataloader, colour="blue", desc="Validation")
            for batch_idx, (images, masks) in enumerate(val_progress):
                images = images.float().to(self.device)
                masks = masks.long().to(self.device)

                # Forward pass
                pred = self.model(images)  # [B, C, H, W]

                # Compute loss
                loss = criterion(pred, masks)

                # Compute metrics
                pred_probs = torch.softmax(pred, dim=1)
                dice = metrics.compute_dice_score(pred_probs, masks)
                iou = metrics.compute_iou(pred_probs, masks)

                epoch_loss += loss.item()
                epoch_dice += dice
                epoch_iou += iou
                num_batches += 1

                msg = (
                    f"Validation | Loss: {epoch_loss / num_batches:.4f} | "
                    f"Dice: {epoch_dice / num_batches:.4f} | "
                    f"IoU: {epoch_iou / num_batches:.4f}"
                )

                val_progress.set_description(msg)

        return {
            "loss": epoch_loss / num_batches,
            "dice": epoch_dice / num_batches,
            "iou": epoch_iou / num_batches,
        }

    def train(
        self,
        epochs: int = 50,
        learning_rate: float = 1e-4,
    ):
        # Initialize optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = MultiClassCombinedLoss(num_classes=3)

        # Metrics
        metrics = MultiClassSegmentationMetrics(num_classes=3)

        # Build Dataloaders
        train_dataloader, val_dataloader = self.__build_dataloaders()

        # Best validation IoU for saving the best model
        best_iou = -1.0
        history = {
            "train_loss": [],
            "train_dice": [],
            "train_iou": [],
            "val_loss": [],
            "val_dice": [],
            "val_iou": [],
        }

        for epoch in range(epochs):
            # Training Phase
            train_metrics = self.__train_epoch(
                dataloader=train_dataloader,
                criterion=criterion,
                optimizer=optimizer,
                metrics=metrics,
            )

            # Validation Phase
            val_metrics = self.__validate(
                dataloader=val_dataloader,
                criterion=criterion,
                metrics=metrics,
            )

            # Save history
            history["train_loss"].append(train_metrics["loss"])
            history["train_dice"].append(train_metrics["dice"])
            history["train_iou"].append(train_metrics["iou"])

            history["val_loss"].append(val_metrics["loss"])
            history["val_dice"].append(val_metrics["dice"])
            history["val_iou"].append(val_metrics["iou"])

            # Save checkpoint
            checkpoint = {
                "model_state_dict": self.model.state_dict(),
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": val_metrics["iou"],
            }

            # Save Last Checkpoint
            torch.save(checkpoint, os.path.join(self.model_path, "last_multiclass.pth"))

            # Save best checkpoint based on IoU
            if val_metrics["iou"] > best_iou:
                torch.save(checkpoint, os.path.join(self.model_path, "best_multiclass.pth"))
                best_iou = val_metrics["iou"]

        return history
