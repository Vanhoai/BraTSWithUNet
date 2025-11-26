import os
import cv2
import torch
from typing import List, Tuple
from torch.utils.data import Dataset


class OxfordIIIPetDataset(Dataset):
    def __init__(
        self,
        root: str,
        is_train: bool = True,
        transform=None,
    ):
        self.root = root
        self.transform = transform
        self.classes = ["background", "animal"]
        self.image_names: List[str] = []

        if is_train:
            annotations = os.path.join(root, "annotations", "trainval.txt")
        else:
            annotations = os.path.join(root, "annotations", "test.txt")

        # Read the annotation file and extract image names
        with open(annotations, "r") as f:
            self.image_names = [image.split(' ')[0] for image in f.readlines()]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        image_name = self.image_names[item]
        image_path = os.path.join(self.root, "images", image_name + ".jpg")
        mask_path = os.path.join(self.root, "annotations", "trimaps", image_name + ".png")

        # Read the image and convert it from BGR to RGB
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read the mask and adjust its values
        # 0.299 x Red + 0.587 x Green + 0.114 x Blue
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask[mask == 2] = 0  # Set background class to 0
        mask[mask == 3] = 1  # Set animal class to 1

        # Apply transformations if provided
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return image, mask
