import os
import torch
from training import OxfordIIITPetTraining

device = torch.accelerator.current_accelerator() or "cpu"

if __name__ == "__main__":
    saved_directory = os.getcwd() + "/saved"
    entire = OxfordIIITPetTraining(saved_directory=saved_directory, device=device)
    entire.train()
