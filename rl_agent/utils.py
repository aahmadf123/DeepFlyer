import numpy as np
import torch
import random

def linear_schedule(initial_value: float):
    """
    Returns a function that computes a linearly decreasing schedule.
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def to_numpy(x):
    """Convert a PyTorch tensor or other to a NumPy array."""
    if hasattr(x, 'detach'):
        x = x.detach()
    if hasattr(x, 'cpu'):
        x = x.cpu()
    return np.array(x)

def set_seed(seed: int):
    """
    Set seeds for random, NumPy, and PyTorch (both CPU and CUDA).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
