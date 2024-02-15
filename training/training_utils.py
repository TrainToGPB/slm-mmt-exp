import os
import random

import torch
import numpy as np


def set_seed(SEED=42):
    """
    Set the random seeds for reproducibility in a PyTorch environment.

    Parameters:
    - SEED (int, optional): Seed value to be used for random number generation. Default is 42.
    """
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(SEED)
