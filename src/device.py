"""
Training device: CUDA if present, else Apple-silicon GPU (MPS), else CPU.

AMP stays CUDA-only in the training scripts (`use_amp = device.type == "cuda"`),
so MPS trains in plain fp32 -- still ~5x faster than the CPU on an M-series Mac.
"""

import torch


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
