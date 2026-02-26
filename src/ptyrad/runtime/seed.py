"""
Random seed resolution and initialization.

This module provides utilities to ensure reproducibility across PtyRAD 
reconstructions by managing random seeds for Python, NumPy, and PyTorch. 
It includes logic to prioritize user-provided seeds and automatically 
enforce seeding during distributed (multi-GPU) runs to prevent divergent 
model initializations.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

def resolve_seed_priority(args_seed, params_seed, acc):
    """Determines the final random seed to use based on user inputs and runtime context.

    This function resolves the seed using the following priority hierarchy:
    
    1. Command Line Argument (`args_seed`)
    2. Configuration File (`params_seed`)
    3. Automatic Fallback (forces seed to 42 if running in multi-GPU mode to 
       ensure consistent probe/object initializations across processes).
    4. None (no fixed seed).

    Args:
        args_seed (int or None): The seed provided via command-line arguments.
        params_seed (int or None): The seed provided in the parameters YAML file.
        acc (accelerate.Accelerator or None): The active HuggingFace Accelerator 
            instance, used to check the number of active processes.

    Returns:
        int or None: The resolved integer seed, or None if no seeding is required.
    """
    
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
    """Sets the random seeds for Python, NumPy, and PyTorch operations.

    Ensures that pseudo-random number generators (PRNGs) across all relevant 
    libraries and devices (CPU and all available CUDA GPUs) are synchronized 
    for reproducible reconstructions.

    Args:
        seed (int, optional): The integer seed to apply. If None, this function 
            does nothing.
        deterministic (bool, optional): If True, forces PyTorch to use deterministic 
            algorithms. Note that this can significantly impact performance and 
            may cause crashes if certain operations do not have a deterministic 
            implementation. Defaults to False.
            
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