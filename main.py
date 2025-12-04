import os

import torch

from models import UNetBaseline
from training import OxfordIIITPetTraining

root = os.getcwd()
device = "mps" if torch.mps.is_available() else "cpu"


def train_binary_oxford_iiit_pet():
    model = UNetBaseline(in_channels=3, num_classes=1).to(device)

    trainer = OxfordIIITPetTraining(
        root=root,
        device=device,
        model=model,
        batch_size=8,
        num_workers=8,
        is_binary=True,
    )

    epochs = 10
    learning_rate = 1e-4
    trainer.train(epochs=epochs, learning_rate=learning_rate)


def train_oxford_iiit_pet():
    model = UNetBaseline(in_channels=3, num_classes=3).to(device)

    trainer = OxfordIIITPetTraining(
        root=root,
        device=device,
        model=model,
        batch_size=8,
        num_workers=8,
        is_binary=False,
    )

    epochs = 10
    learning_rate = 1e-4
    trainer.train(epochs=epochs, learning_rate=learning_rate)


if __name__ == "__main__":
    train_oxford_iiit_pet()
