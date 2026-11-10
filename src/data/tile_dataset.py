"""
(LR, HR) pairs cut directly from the raw tiles, using a manifest from
src/data/spatial_splits.py.

train: a fresh random crop on every __getitem__, drawn from the train bands in
proportion to their area. A run therefore sees ~len x epochs distinct crops
rather than the same fixed patches. Crops that are cloudy (per the tile's
mask) or flat are redrawn, using the same thresholds as patchify.py.

val / test: a fixed, non-overlapping grid of crops inside each band, filtered
the same way, with LR noise seeded by index -- so evaluation is repeatable.

Same output as SatelliteSRDataset: tensors (4, 32, 32) and (4, 128, 128).

Usage:
    train_loader, val_loader = get_tile_dataloaders(batch_size=8)
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.degrade import degrade


class TileSRDataset(Dataset):
    def __init__(self, split, manifest_path="data/splits/manifest_spatial.json",
                 augment=None, length=2456, crop_size=128, scale_factor=4,
                 blur_sigma=1.0, noise_std=0.01, max_bad_frac=0.01, min_std=0.002,
                 max_redraws=20):
        if not Path(manifest_path).exists():
            raise FileNotFoundError(
                f"{manifest_path} not found. Build it first:\n"
                f"    python -m src.data.spatial_splits --mode spatial")
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.split = split
        self.random_crops = split == "train"
        self.augment = self.random_crops if augment is None else augment
        self.length = length
        self.crop = crop_size
        self.scale_factor = scale_factor
        self.blur_sigma = blur_sigma
        self.noise_std = noise_std
        self.max_bad_frac = max_bad_frac
        self.min_std = min_std
        self.max_redraws = max_redraws
        self.tiles_dir = Path(manifest["tiles_dir"])
        self._tiles, self._masks = {}, {}

        self.bands = [b for b in manifest["splits"][split]
                      if b["y1"] - b["y0"] >= crop_size and b["x1"] - b["x0"] >= crop_size]
        if not self.bands:
            raise ValueError(f"No '{split}' band in {manifest_path} fits a {crop_size}px crop")

        if self.random_crops:
            areas = np.array([(b["y1"] - b["y0"]) * (b["x1"] - b["x0"]) for b in self.bands],
                             dtype=np.float64)
            self.band_weights = torch.from_numpy(areas / areas.sum())
        else:
            self.crops = [(b["tile"], y, x) for b in self.bands
                          for y in range(b["y0"], b["y1"] - crop_size + 1, crop_size)
                          for x in range(b["x0"], b["x1"] - crop_size + 1, crop_size)
                          if self._usable(b["tile"], y, x)]
            if not self.crops:
                raise ValueError(f"Every '{split}' crop was cloudy or flat")

    # tiles are memory-mapped lazily so each DataLoader worker opens its own
    def _tile(self, name):
        if name not in self._tiles:
            self._tiles[name] = np.load(self.tiles_dir / f"{name}.npy", mmap_mode="r")
            mask_path = self.tiles_dir / "masks" / f"{name}.npy"
            self._masks[name] = np.load(mask_path, mmap_mode="r") if mask_path.exists() else None
        return self._tiles[name], self._masks[name]

    def _read(self, name, y, x):
        tile, _ = self._tile(name)
        return np.asarray(tile[y:y + self.crop, x:x + self.crop, :4], dtype=np.float32)

    def _usable(self, name, y, x, patch=None):
        _, mask = self._tile(name)
        if mask is not None and mask[y:y + self.crop, x:x + self.crop].mean() > self.max_bad_frac:
            return False
        patch = self._read(name, y, x) if patch is None else patch
        return bool(np.isfinite(patch).all() and patch.std() >= self.min_std)

    def __len__(self):
        return self.length if self.random_crops else len(self.crops)

    def _random_crop(self):
        for _ in range(self.max_redraws):
            b = self.bands[torch.multinomial(self.band_weights, 1).item()]
            y = torch.randint(b["y0"], b["y1"] - self.crop + 1, (1,)).item()
            x = torch.randint(b["x0"], b["x1"] - self.crop + 1, (1,)).item()
            patch = self._read(b["tile"], y, x)
            if self._usable(b["tile"], y, x, patch):
                return patch
        return patch   # a run of bad draws; one imperfect crop is harmless

    def __getitem__(self, idx):
        if self.random_crops:
            hr = self._random_crop()
        else:
            hr = self._read(*self.crops[idx])

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


def get_tile_dataloaders(batch_size=8, manifest_path="data/splits/manifest_spatial.json",
                         train_length=2456, num_workers=2):
    train_ds = TileSRDataset("train", manifest_path=manifest_path, length=train_length)
    val_ds = TileSRDataset("val", manifest_path=manifest_path, augment=False)
    # train crops are already random, so shuffling the indices adds nothing
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader
