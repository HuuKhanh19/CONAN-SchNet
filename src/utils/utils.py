"""Utility functions for CONAN-SchNet."""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int):
    """Set all random seeds for reproducibility.
    
    Controls:
        - Python's random module
        - PYTHONHASHSEED environment variable
        - NumPy's global random state
        - PyTorch CPU & CUDA random states
        - cuDNN deterministic mode
    
    Note:
        When using DataLoader with num_workers > 0, each worker needs
        its own seed via worker_init_fn. See get_worker_init_fn().
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_worker_init_fn(base_seed: int):
    """Return a worker_init_fn for DataLoader reproducibility.
    
    Usage:
        DataLoader(..., worker_init_fn=get_worker_init_fn(42))
    """
    def worker_init_fn(worker_id: int):
        worker_seed = base_seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
    return worker_init_fn