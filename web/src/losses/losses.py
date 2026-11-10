def compute_ndvi(img, red_idx=0, nir_idx=3, eps=1e-6):
    """
    Computes Normalized Difference Vegetation Index (NDVI) map.
    img: (B, C, H, W) tensor where bands are [Red, Green, Blue, NIR]
    NDVI = (NIR - Red) / (NIR + Red + eps)
    """
    red = img[:, red_idx:red_idx+1, :, :]
    nir = img[:, nir_idx:nir_idx+1, :, :]
    return (nir - red) / (nir + red + eps)
