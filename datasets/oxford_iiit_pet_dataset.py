import os
import warnings
from typing import List

import cv2
import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)


class OxfordIIIPetDataset(Dataset):
    def __init__(
        self,
        root: str,
        is_train: bool = True,
        transform=None,
    ):
        self.root = root
        self.transform = transform
        self.classes = ["background", "animal", "border"]
        self.image_names: List[str] = []

        if is_train:
            annotations = os.path.join(root, "annotations", "trainval.txt")
        else:
            annotations = os.path.join(root, "annotations", "test.txt")

        # Read the annotation file and extract image names
        with open(annotations, "r") as f:
            self.image_names = [image.split(" ")[0] for image in f.readlines()]

        if not self.image_names:
            raise ValueError(f"No images found in the dataset at {annotations}")

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        try:
            image_name = self.image_names[idx]
            image_path = os.path.join(self.root, "images", image_name + ".jpg")
            mask_path = os.path.join(
                self.root,
                "annotations",
                "trimaps",
                image_name + ".png",
            )

            image_pil = Image.open(image_path).convert("RGB")

            # Convert to numpy array (RGB format)
            image = np.array(image_pil)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Cannot load mask: {mask_path}")

            # Convert mask values from {1,2,3} to {0,1,2}
            mask = mask - 1

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]

            return image, mask

        except Exception as exception:
            print(f"Loading exception at index {idx}: {exception}")
            next_idx = (idx + 1) % len(self.image_names)
            return self.__getitem__(next_idx)
