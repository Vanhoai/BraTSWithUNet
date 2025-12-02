import os
import torch
import numpy as np
import nibabel as nib
from typing import List, Tuple
from torch.utils.data import Dataset
import cv2

SEGMENT_CLASSES = {
    0: "NOT tumor",
    1: "NECROTIC/CORE",  # or NON-ENHANCING tumor CORE
    2: "EDEMA",
    3: "ENHANCING",  # original 4 -> converted into 3
}

# Select Slices and Image Size
VOLUME_SLICES = 100
VOLUME_START_AT = 22  # first slice of volume that we will include
IMG_SIZE = 128


class BrainTumorDataset(Dataset):
    def __init__(
        self,
        ids: List[str],
        data_path: str,
        img_size: int = IMG_SIZE,
        n_channels: int = 2,
        volume_slices: int = VOLUME_SLICES,
        volume_start_at: int = VOLUME_START_AT,
        transform=None,
    ) -> None:
        self.ids = ids
        self.data_path = data_path
        self.img_size = img_size
        self.n_channels = n_channels
        self.volume_slices = volume_slices
        self.volume_start_at = volume_start_at
        self.transform = transform

        # Calculate total number of slices
        self.total_slices = len(self.ids) * self.volume_slices

    def __len__(self) -> int:
        return self.total_slices

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get one slice of data
        Returns:
            X: Input tensor of shape (n_channels, img_size, img_size)
            Y: Target mask tensor of shape (4, img_size, img_size) - one-hot encoded

        Example:
            idx=0
            patient_idx = 0 // 100 = 0
            slice_idx = 0 % 100 = 0
            -> First slice of first patient
        """

        # Calculate which patient and which slice
        patient_idx = idx // self.volume_slices
        slice_idx = idx % self.volume_slices

        patient_id = self.ids[patient_idx]
        case_path = case_path = os.path.join(self.data_path, patient_id)

        # Load FLAIR
        flair_path = os.path.join(case_path, f"{patient_id}_flair.nii")
        flair = nib.load(flair_path).get_fdata()  # type: ignore

        # Load T1CE
        t1ce_path = os.path.join(case_path, f"{patient_id}_t1ce.nii")
        t1ce = nib.load(t1ce_path).get_fdata()  # type: ignore

        # Load Segmentation mask
        seg_path = os.path.join(case_path, f"{patient_id}_seg.nii")
        seg = nib.load(seg_path).get_fdata()  # type: ignore

        # Extract and resize the specific slice
        actual_slice = slice_idx + self.volume_start_at

        flair_slice = cv2.resize(
            flair[:, :, actual_slice],
            (self.img_size, self.img_size),
        )

        t1ce_slice = cv2.resize(
            t1ce[:, :, actual_slice],
            (self.img_size, self.img_size),
        )

        seg_slice = cv2.resize(
            seg[:, :, actual_slice],
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST,  # Use nearest neighbor for labels
        )

        # Stack channels: (H, W, C) -> (C, H, W)
        X = np.stack([flair_slice, t1ce_slice], axis=0).astype(np.float32)

        # Normalize input
        X = X / (np.max(X) + 1e-8)  # Add small epsilon to avoid division by zero

        # Process segmentation mask
        # Convert label 4 to 3
        seg_slice[seg_slice == 4] = 3

        # One-hot encode: (H, W) -> (4, H, W)
        Y = np.zeros((4, self.img_size, self.img_size), dtype=np.float32)
        for c in range(4):
            Y[c] = (seg_slice == c).astype(np.float32)

        # Convert to tensors
        X = torch.from_numpy(X)
        Y = torch.from_numpy(Y)

        # Apply transforms if any
        if self.transform:
            X, Y = self.transform(X, Y)

        return X, Y

    def get_patient_volume(self, patient_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get all slices for a specific patient

        Returns:
            X: Input tensor of shape (volume_slices, n_channels, img_size, img_size)
            Y: Target mask tensor of shape (volume_slices, 4, img_size, img_size)
        """
        start_idx = patient_idx * self.volume_slices
        end_idx = start_idx + self.volume_slices

        X_list = []
        Y_list = []

        for idx in range(start_idx, end_idx):
            X, Y = self[idx]
            X_list.append(X)
            Y_list.append(Y)

        X_volume = torch.stack(X_list, dim=0)
        Y_volume = torch.stack(Y_list, dim=0)

        return X_volume, Y_volume
