import logging
from typing import Optional

logger = logging.getLogger(__name__)

def resolve_seed_priority(args_seed, params_seed, acc):
    
    if  args_seed is not None:
        seed = args_seed
        logger.info(f"Random seed: {seed} provided by CLI argument")
    elif  params_seed is not None:
        seed = params_seed
        logger.info(f"Random seed: {seed} provided by params file")
    elif acc is not None and acc.num_processes > 1:
        seed = 42 # seed is required otherwise the probe position with random displacement could cause objects with different shapes
        logger.info(f"Random seed: {seed} is set automatically because multi GPU is detected but no seed is provided")
    else:
        seed = None
    return seed

def set_random_seed(seed: Optional[int], deterministic: bool = False):
    """
    Set the random seeds for numpy and pytorch operations.
    
    """
    
    import random

    import numpy as np
    import torch
    
    if seed is not None:
        random.seed(seed)                  # Python's RNG
        np.random.seed(seed)               # NumPy RNG
        torch.manual_seed(seed)            # PyTorch CPU RNG
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed) # All GPUs
        
        if deterministic:
            # This would slow down the operation a bit: https://docs.pytorch.org/docs/stable/notes/randomness.html
            torch.use_deterministic_algorithms(True)