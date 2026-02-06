"""
General purpose utility tools for logging, converting types, parsing dicts and strings, etc.

"""



from typing import Optional

from .logging import vprint


# Only used in run_ptyrad.py, might have a better place
def set_accelerator():

    try:
        import torch
        from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs
        from accelerate.state import DistributedType
        dataloader_config  = DataLoaderConfiguration(split_batches=True) # This supress the warning when we do `Accelerator(split_batches=True)`
        kwargs_handlers    = [DistributedDataParallelKwargs(find_unused_parameters=True)] # This avoids the error `RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss.` Previously we don't necessarily need this if we carefully register parameters (used in forward) and buffer in the `model`. This is now needed if we want to toggle the grad for optimizable tensors dynamically between iterations.
        accelerator        = Accelerator(dataloader_config=dataloader_config, kwargs_handlers=kwargs_handlers)
        vprint("### Initializing HuggingFace accelerator ###")
        vprint(f"Accelerator.distributed_type = {accelerator.distributed_type}")
        vprint(f"Accelerator.num_process      = {accelerator.num_processes}")
        vprint(f"Accelerator.mixed_precision  = {accelerator.mixed_precision}")
        
        # Check if the number of processes exceeds available GPUs
        device_count = max(torch.cuda.device_count(), torch.mps.device_count())
        if accelerator.num_processes > device_count:
            vprint(f"ERROR: The specified number of processes for 'accelerate' ({accelerator.num_processes}) exceeds the number of GPUs available ({device_count}).")
            vprint("Please verify the following:")
            vprint("  1. Check the number of GPUs available on your system with `nvidia-smi` if you're using NVIDIA GPUs.")
            vprint("  2. If using a SLURM cluster, ensure your job script requests the correct number of GPUs (e.g., `--gres=gpu:<num_gpus>`).")
            vprint("  3. Ensure your environment is correctly configured to detect GPUs (e.g., CUDA drivers are installed and compatible).")
            raise ValueError("The number of processes exceeds the available GPUs. Please adjust your configuration.")
        
        if accelerator.distributed_type == DistributedType.NO and accelerator.mixed_precision == "no":
            vprint("'accelerate' is available but NOT using distributed mode or mixed precision")
            vprint("If you want to utilize 'accelerate' for multiGPU or mixed precision, ")
            vprint("Run `accelerate launch --multi_gpu --num_processes=2 --mixed_precision='no' -m ptyrad run <PTYRAD_ARGUMENTS> --gpuid 'acc'` in your terminal")
    except ImportError:
        vprint("### HuggingFace accelerator is not available, no multi-GPU or mixed-precision ###")
        accelerator = None
        
    vprint(" ")
    return accelerator

def set_gpu_device(gpuid=0):
    """
    Sets the GPU device based on the input. If 'acc' is passed, it returns None to defer to accelerate.
    
    Args:
        gpuid (str or int): The GPU ID to use. Can be:
            - "acc": Defer device assignment to accelerate.
            - "cpu": Use CPU.
            - An integer (or string representation of an integer) for a specific GPU ID. This only has effect on NVIDIA GPUs.
    
    Returns:
        torch.device or None: The selected device, or None if deferred to accelerate.
    """
    import torch
    
    vprint("### Setting GPU Device ###")

    if gpuid == "acc":
        vprint("Specified to use accelerate device (gpuid='acc')")
        vprint(" ")
        return None
    
    if gpuid == "cpu":
        device = torch.device("cpu")
        torch.set_default_device(device)
        vprint("Specified to use CPU (gpuid='cpu').")
        vprint(" ")
        return device

    try:
        gpuid = int(gpuid)
        if torch.cuda.is_available():
            num_cuda_devices = torch.cuda.device_count()
            if gpuid < num_cuda_devices:
                device = torch.device(f"cuda:{gpuid}")
                torch.set_default_device(device)
                vprint(f"Selected GPU device: {device} ({torch.cuda.get_device_name(gpuid)})")
                vprint(" ")
                return device
            
            else:
                device = torch.device("cuda")
                vprint(f"Requested CUDA device cuda:{gpuid} is out of range (only {num_cuda_devices} available). " 
                    f"Fall back to GPU device: {device}")
                vprint(" ")
                return device
            
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            torch.set_default_device(device)
            vprint("Selected GPU device: MPS (Apple Silicon)")
            vprint(" ")
            return device
        
        else:
            device = torch.device("cpu")
            torch.set_default_device(device)
            vprint(f"GPU ID specifed as {gpuid} but no GPU found. Using CPU instead.")
            vprint(" ")
            return device
        
    except ValueError:
        raise ValueError(f"Invalid gpuid '{gpuid}'. Expected 'acc', 'cpu', or an integer.")

def resolve_seed_priority(args_seed, params_seed, acc):
    
    vprint("### Resolving random seed ###")
    if  args_seed is not None:
        seed = args_seed
        vprint(f"Random seed: {seed} provided by CLI argument")
    elif  params_seed is not None:
        seed = params_seed
        vprint(f"Random seed: {seed} provided by params file")
    elif acc is not None and acc.num_processes > 1:
        seed = 42 # seed is required otherwise the probe position with random displacement could cause objects with different shapes
        vprint(f"Random seed: {seed} is set automatically because multi GPU is detected but no seed is provided")
    else:
        seed = None
    vprint(" ")
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
    

def get_nested(d, key, delimiter='.', safe=False, default=None):
    """
    Get a value from a nested dictionary either safely (return default if not found) or stricly to fail early.
    
    Parameters:
    - d (dict): The dictionary to traverse.
    - key (str, or list or tuple of string): A sequence of keys to access nested values.
    - delimiter (str): The string used to seperate different parts of the displayed key path
    - safe (boolean): The flag to switch between safe/strict mode of getting values from a nested dict.
    - default: The value to return if any key is missing or intermediate value is None.
    
    Returns:
    - The nested value if found, otherwise `default` in safe mode or error in strict mode.
    """
    
    if not key:
        raise ValueError("Please specify a non-empty 'key' to get the value from a nested dict.")

    # Parse the input key (str with delimiter, or sequence of strings)
    if isinstance(key, str):
        parts = key.split(delimiter)
    elif isinstance(key, (tuple, list)):
        if not all(isinstance(k, str) for k in key):
            raise TypeError(
                f"All elements in 'key' must be strings, got {[type(k).__name__ for k in key]}"
            )
        parts = key
    else:
        raise TypeError(f"'key' must be a str, or a sequence (list, tuple) of strings, got {type(key).__name__}.")
    
    # Getting value safely with a default return 
    if safe:
        for k in parts:
            if not isinstance(d, dict):
                return default
            d = d.get(k)
            if d is None:
                return default
        return d
    
    # Getting value strictly with raised error
    else:
        for k in parts:
            if not isinstance(d, dict) or k not in d:
                raise KeyError(
                    f"Key '{key}' not found. Failed at '{k}'. "
                    f"Available key(s) in this nested dict are {list_nested_keys(d)}. "
                    "Tip: If you don't know the correct key, use `vprint_nested_dict()` from `ptyrad.utils.common` to check your nested dict first."
                )
            d = d[k]
        return d
    
def list_nested_keys(hobj, delimiter=".", prefix=""):
    """
    Recursively list all keys in an HDF5 file, HDF5 group, or dict, including hierarchical paths.

    Args:
        hobj (h5py.File, h5py.Group, or dict): The hierarchical object to traverse.
        delimiter (str): The string used to seperate different parts of the displayed key path
        prefix (str): The current hierarchical path (used for recursion).

    Returns:
        list[str]: A list of all keys with their hierarchical paths.
    """
    import h5py
    
    # Check input type
    if isinstance(hobj, (h5py.Group, h5py.File)):
        compare_type = h5py.Group
    elif isinstance(hobj, dict):
        compare_type = dict
    else:
        raise ValueError(f"Expected hobj is an HDF5 file, HDF5 group, or a dict, got {type(hobj).__name__}.")
    
    keys = []
    for key in hobj.keys():
        full_key = f"{prefix}{key}" if prefix == "" else f"{prefix}{delimiter}{key}"
        if isinstance(hobj[key], compare_type):
            # Recursively list keys in the group / dict
            keys.extend(list_nested_keys(hobj[key], delimiter=delimiter, prefix=full_key))
        else:
            # Add dataset key
            keys.append(full_key)
    return keys
