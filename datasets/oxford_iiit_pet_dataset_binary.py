import os
import warnings
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)


class OxfordIIIPetBinaryDataset(Dataset):
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
            self.image_names = [image.split(" ")[0] for image in f.readlines()]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            image_name = self.image_names[item]
            image_path = os.path.join(self.root, "images", image_name + ".jpg")
            mask_path = os.path.join(
                self.root,
                "annotations",
                "trimaps",
                image_name + ".png",
            )

            image_pil = Image.open(image_path).convert("RGB")
            image = np.array(image_pil)  # Convert to numpy array (RGB format)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                raise ValueError(f"Cannot load mask: {mask_path}")

            mask[mask == 2] = 0  # type: ignore
            mask[mask == 3] = 1  # type: ignore

            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]

            return image, mask  # type: ignore

        except Exception as e:
            print(
                f"Error loading image at index {item} ({self.image_names[item]}): {e}"
            )
            next_idx = (item + 1) % len(self.image_names)
            return self.__getitem__(next_idx)
