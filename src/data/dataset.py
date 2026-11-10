"""
Module 3: (LR, HR) pairs for training and evaluation.

HR is a stored 128x128x4 patch; LR is made on the fly with degrade(), so a
training run sees fresh sensor noise every epoch. With augment=False the
noise is seeded by the patch index, so evaluation numbers are repeatable.

Returns tensors shaped (4, 32, 32) and (4, 128, 128), bands R, G, B, NIR.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.degrade import degrade


class SatelliteSRDataset(Dataset):
    def __init__(self, patches_dir="data/patches", scale_factor=4, augment=True,
                 blur_sigma=1.0, noise_std=0.01):
        self.patch_files = sorted(Path(patches_dir).glob("*.npy"))
        if not self.patch_files:
            raise FileNotFoundError(
                f"No patches in {patches_dir}/ -- run `python -m src.data.patchify` first.")
        self.scale_factor = scale_factor
        self.augment = augment
        self.blur_sigma = blur_sigma
        self.noise_std = noise_std

    def __len__(self):
        return len(self.patch_files)

    def __getitem__(self, idx):
        hr = np.load(self.patch_files[idx]).astype(np.float32)

        if self.augment:
            # torch's RNG is seeded per DataLoader worker; numpy's global one is not
            k, flip_h, flip_v, seed = torch.randint(0, 2 ** 31 - 1, (4,)).tolist()
            hr = np.rot90(hr, k % 4, axes=(0, 1))
            if flip_h % 2:
                hr = hr[:, ::-1]
            if flip_v % 2:
                hr = hr[::-1, :]
            rng = np.random.default_rng(seed)
        else:
            rng = np.random.default_rng(idx)

        hr = np.ascontiguousarray(hr)
        lr = degrade(hr, self.scale_factor, self.blur_sigma, self.noise_std, rng=rng)
        return (torch.from_numpy(lr.transpose(2, 0, 1).copy()),
                torch.from_numpy(hr.transpose(2, 0, 1).copy()))
