# import os
# import torch
# from training import OxfordIIITPetTraining


# def train_oxford_iiit_pet_model():
#     device = torch.accelerator.current_accelerator() or "cpu"
#     saved_directory = os.getcwd() + "/saved"
#     entire = OxfordIIITPetTraining(saved_directory=saved_directory, device=device)  # type: ignore
#     entire.train()

import torch

from models import AttentionUNet


def train_attention_unet():
    model = AttentionUNet(in_channels=3, out_channels=1)
    x = torch.randn((1, 3, 256, 256))
    out = model(x)
    print(out.shape)


if __name__ == "__main__":
    # train_oxford_iiit_pet_model()
    train_attention_unet()
