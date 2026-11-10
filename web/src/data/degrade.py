"""HR -> LR degradation used by the web inference path.

Matches the training setup described in the project README and
web/configs/config.yaml: Gaussian blur (sigma 1.0), x4 subsampling,
then a small amount of sensor noise.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def degrade(hr, scale_factor=4, blur_sigma=1.0, noise_std=0.01):
    """Downsample an HR tile (H, W, C) to a low-resolution input."""
    hr = np.asarray(hr, dtype=np.float32)
    if hr.ndim != 3:
        raise ValueError(f"Expected (H, W, C) tile, got shape {hr.shape}")
    if scale_factor < 1:
        raise ValueError("scale_factor must be >= 1")

    blurred = gaussian_filter(hr, sigma=(blur_sigma, blur_sigma, 0.0))
    lr = blurred[::scale_factor, ::scale_factor, :]
    if noise_std > 0:
        lr = lr + np.random.normal(0.0, noise_std, size=lr.shape).astype(np.float32)
    return np.clip(lr, 0.0, 1.0).astype(np.float32)
