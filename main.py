import os

import torch

from models import UNetBaseline
from training import OxfordIIITPetTraining

if __name__ == "__main__":
    root = os.getcwd()
    device = "mps" if torch.mps.is_available() else "cpu"
    model = UNetBaseline(in_channels=3, num_classes=1).to(device)

    trainer = OxfordIIITPetTraining(root=root, device=device, model=model)

    epochs = 10
    learning_rate = 1e-4
    trainer.train(epochs=epochs, learning_rate=learning_rate)
