"""
Runtime environment and hardware configuration.

This module handles the initialization of PyTorch device settings and 
HuggingFace Accelerate environments, enabling seamless transitions 
between single-GPU, multi-GPU, CPU, and Apple Silicon (MPS) runtimes.
"""

from typing import Literal, Optional, Union
import logging

logger = logging.getLogger(__name__)

def set_accelerator():
    """Initializes the HuggingFace Accelerator for distributed training.

    This function attempts to load `accelerate` and configure it for PtyRAD's 
    specific needs. It enables `split_batches=True` for data loaders and sets 
    `find_unused_parameters=True` for DistributedDataParallel (DDP) to prevent 
    crashes when toggling gradients dynamically between iterations. It also 
    verifies that the requested number of processes does not exceed the available 
    hardware.

    Returns:
        accelerate.Accelerator or None: The initialized Accelerator object if 
        the library is available and correctly configured, otherwise None.

    Raises:
        ValueError: If the configured number of accelerate processes exceeds 
            the actual number of available GPUs on the system.
    """
    try:
        import torch
        from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs
        from accelerate.state import DistributedType
        dataloader_config  = DataLoaderConfiguration(split_batches=True) # This supress the warning when we do `Accelerator(split_batches=True)`
        kwargs_handlers    = [DistributedDataParallelKwargs(find_unused_parameters=True)] # This avoids the error `RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss.` Previously we don't necessarily need this if we carefully register parameters (used in forward) and buffer in the `model`. This is now needed if we want to toggle the grad for optimizable tensors dynamically between iterations.
        accelerator        = Accelerator(dataloader_config=dataloader_config, kwargs_handlers=kwargs_handlers)
        logger.info("### Initializing HuggingFace accelerator ###")
        logger.info(f"Accelerator.distributed_type = {accelerator.distributed_type}")
        logger.info(f"Accelerator.num_process      = {accelerator.num_processes}")
        logger.info(f"Accelerator.mixed_precision  = {accelerator.mixed_precision}")
        
        # Check if the number of processes exceeds available GPUs
        device_count = max(torch.cuda.device_count(), torch.mps.device_count())
        if accelerator.num_processes > device_count:
            logger.info(f"ERROR: The specified number of processes for 'accelerate' ({accelerator.num_processes}) exceeds the number of GPUs available ({device_count}).")
            logger.info("Please verify the following:")
            logger.info("  1. Check the number of GPUs available on your system with `nvidia-smi` if you're using NVIDIA GPUs.")
            logger.info("  2. If using a SLURM cluster, ensure your job script requests the correct number of GPUs (e.g., `--gres=gpu:<num_gpus>`).")
            logger.info("  3. Ensure your environment is correctly configured to detect GPUs (e.g., CUDA drivers are installed and compatible).")
            raise ValueError("The number of processes exceeds the available GPUs. Please adjust your configuration.")
        
        if accelerator.distributed_type == DistributedType.NO and accelerator.mixed_precision == "no":
            logger.info("'accelerate' is available but NOT using distributed mode or mixed precision")
            logger.info("If you want to utilize 'accelerate' for multiGPU or mixed precision, ")
            logger.info("Run `accelerate launch --multi_gpu --num_processes=2 --mixed_precision='no' -m ptyrad run <PTYRAD_ARGUMENTS> --gpuid 'acc'` in your terminal")
    except ImportError:
        logger.info("### HuggingFace accelerator is not available, no multi-GPU or mixed-precision ###")
        accelerator = None
        
    logger.info(" ")
    return accelerator

def set_gpu_device(gpuid: Optional[Union[Literal["acc", "cpu"], int]] = 0):
    """Sets the default PyTorch computation device.

    If 'acc' is passed, device management is explicitly deferred to the 
    HuggingFace Accelerator. Otherwise, it configures the global PyTorch 
    default device to the requested CUDA GPU, Apple Silicon (MPS), or CPU, 
    handling fallbacks automatically if the requested hardware is unavailable.
    
    Args:
        gpuid (str or int, optional): The device identifier to use. Can be:
        
            * "acc": Defer device assignment to accelerate.
            * "cpu": Force execution on the CPU.
            * An integer (or string representation of an integer) for a 
              specific NVIDIA GPU ID. Defaults to 0.
    
    Returns:
        torch.device or None: The selected PyTorch device object, or None if 
        assignment was deferred to accelerate.

    Raises:
        ValueError: If `gpuid` cannot be parsed as 'acc', 'cpu', or an integer.
    """
    import torch
    
    logger.info("### Setting GPU Device ###")

    if gpuid == "acc":
        logger.info("Specified to use accelerate device (gpuid='acc')")
        logger.info(" ")
        return None
    
    if gpuid == "cpu":
        device = torch.device("cpu")
        torch.set_default_device(device)
        logger.info("Specified to use CPU (gpuid='cpu').")
        logger.info(" ")
        return device

    try:
        gpuid = int(gpuid)
        if torch.cuda.is_available():
            num_cuda_devices = torch.cuda.device_count()
            if gpuid < num_cuda_devices:
                device = torch.device(f"cuda:{gpuid}")
                torch.set_default_device(device)
                logger.info(f"Selected GPU device: {device} ({torch.cuda.get_device_name(gpuid)})")
                logger.info(" ")
                return device
            
            else:
                device = torch.device("cuda")
                logger.info(f"Requested CUDA device cuda:{gpuid} is out of range (only {num_cuda_devices} available). " 
                    f"Fall back to GPU device: {device}")
                logger.info(" ")
                return device
            
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            torch.set_default_device(device)
            logger.info("Selected GPU device: MPS (Apple Silicon)")
            logger.info(" ")
            return device
        
        else:
            device = torch.device("cpu")
            torch.set_default_device(device)
            logger.info(f"GPU ID specifed as {gpuid} but no GPU found. Using CPU instead.")
            logger.info(" ")
            return device
        
    except ValueError:
        raise ValueError(f"Invalid gpuid '{gpuid}'. Expected 'acc', 'cpu', or an integer.")