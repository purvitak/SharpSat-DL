"""
Module 3: HR -> LR degradation. Gaussian blur (sigma 1.0), x4 subsampling,
then additive sensor noise (std 0.01) -- the settings in configs/config.yaml.

The untouched HR patch is the ground truth the model is scored against.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def degrade(hr, scale_factor=4, blur_sigma=1.0, noise_std=0.01, rng=None):
    """hr: (H, W, C) float reflectance in [0, 1]. Returns (H/s, W/s, C) float32."""
    hr = np.asarray(hr, dtype=np.float32)
    if hr.ndim != 3:
        raise ValueError(f"Expected (H, W, C), got shape {hr.shape}")
    blurred = gaussian_filter(hr, sigma=(blur_sigma, blur_sigma, 0.0))
    lr = blurred[::scale_factor, ::scale_factor, :]
    if noise_std > 0:
        rng = rng if rng is not None else np.random.default_rng()
        lr = lr + rng.normal(0.0, noise_std, size=lr.shape).astype(np.float32)
    return np.clip(lr, 0.0, 1.0).astype(np.float32)
