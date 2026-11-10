"""Sample tiles for the web dashboard.

Real Sentinel tiles and checkpoints are not stored in git. When
data/raw_tiles is empty, write a small synthetic 4-band scene so the
preset picker has something to load.
"""

from pathlib import Path

import numpy as np


def _synthetic_tile(height=256, width=256, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    coast = 0.32 + 0.08 * np.sin(yy / 22.0)
    land = (xx / width) > coast
    veg = land & (((xx // 16 + yy // 16) % 2) == 0)

    red = np.where(land, np.where(veg, 0.09, 0.28), 0.03)
    green = np.where(land, np.where(veg, 0.20, 0.22), 0.07)
    blue = np.where(land, np.where(veg, 0.06, 0.14), 0.20)
    nir = np.where(veg, 0.48, np.where(land, 0.18, 0.02))
    tile = np.stack([red, green, blue, nir], axis=-1).astype(np.float32)
    tile += rng.normal(0.0, 0.008, size=tile.shape).astype(np.float32)
    return np.clip(tile, 0.0, 1.0).astype(np.float32)


def download_or_generate_dataset(raw_tiles_dir="data/raw_tiles"):
    tiles_dir = Path(raw_tiles_dir)
    tiles_dir.mkdir(parents=True, exist_ok=True)
    existing = list(tiles_dir.glob("*.npy"))
    if existing:
        return existing

    out = tiles_dir / "demo_scene.npy"
    np.save(out, _synthetic_tile())
    return [out]
