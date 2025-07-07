"""
General purpose utility tools for logging, converting types, parsing dicts and strings, etc.

"""

import io
import logging
import os
import platform
import subprocess
import warnings
from time import perf_counter
from typing import Union

import numpy as np
import torch
import torch.distributed as dist


def is_mig_enabled():
    """
    Detects if any GPU on the system is operating in MIG (Multi-Instance GPU) mode.
    
    Returns:
        bool: True if MIG mode is enabled on any GPU, False otherwise.
    """
    try:
        # Run the `nvidia-smi` command to query MIG mode
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=mig.mode.current", "--format=csv,noheader"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        # Check for errors in the command execution
        if result.returncode != 0:
            print(f"Error running nvidia-smi: {result.stderr.strip()}")
            return False
        
        # Parse the output to check for MIG mode
        mig_modes = result.stdout.strip().split("\n")
        for mode in mig_modes:
            if mode.strip() == "Enabled":
                return True
        
        return False
    except FileNotFoundError:
        # `nvidia-smi` is not available
        print("nvidia-smi not found. Unable to detect MIG mode.")
        return False
    except Exception as e:
        # Catch other unexpected errors
        print(f"Error detecting MIG mode: {e}")
        return False

# Only used in run_ptyrad.py, might have a better place
def set_accelerator():

    try:
        from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs
        from accelerate.state import DistributedType
        dataloader_config  = DataLoaderConfiguration(split_batches=True) # This supress the warning when we do `Accelerator(split_batches=True)`
        kwargs_handlers    = [DistributedDataParallelKwargs(find_unused_parameters=False)] # This avoids the error `RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. This error indicates that your module has parameters that were not used in producing loss.` We don't necessarily need this if we carefully register parameters (used in forward) and buffer in the `model`.
        accelerator        = Accelerator(dataloader_config=dataloader_config, kwargs_handlers=kwargs_handlers)
        vprint("### Initializing HuggingFace accelerator ###")
        vprint(f"Accelerator.distributed_type = {accelerator.distributed_type}")
        vprint(f"Accelerator.num_process      = {accelerator.num_processes}")
        vprint(f"Accelerator.mixed_precision  = {accelerator.mixed_precision}")
        
        # Check if the number of processes exceeds available GPUs
        if accelerator.num_processes > torch.cuda.device_count():
            vprint(f"ERROR: The specified number of processes for 'accelerate' ({accelerator.num_processes}) exceeds the number of GPUs available ({torch.cuda.device_count()}).")
            vprint("Please verify the following:")
            vprint("  1. Check the number of GPUs available on your system using `nvidia-smi`.")
            vprint("  2. If using a SLURM cluster, ensure your job script requests the correct number of GPUs (e.g., `--gres=gpu:<num_gpus>`).")
            vprint("  3. Ensure your environment is correctly configured to detect GPUs (e.g., CUDA drivers are installed and compatible).")
            raise ValueError("The number of processes exceeds the available GPUs. Please adjust your configuration.")
        
        if accelerator.distributed_type == DistributedType.NO and accelerator.mixed_precision == "no":
            vprint("'accelerate' is available but NOT using distributed mode or mixed precision")
            vprint("If you want to utilize 'accelerate' for multiGPU or mixed precision, ")
            vprint("Run `accelerate launch --multi_gpu --num_processes=2 --mixed_precision='no' -m ptyrad run <PTYRAD_ARGUMENTS>` in your terminal")
    except ImportError:
        vprint("### HuggingFace accelerator is not available, no multi-GPU or mixed-precision ###")
        accelerator = None
        
    vprint(" ")
    return accelerator

# System level utils
class CustomLogger:
    def __init__(self, log_file='ptyrad_log.txt', log_dir='auto', prefix_time='datetime', prefix_date=None, prefix_jobid=0, append_to_file=True, show_timestamp=True, **kwargs):
        self.logger = logging.getLogger('PtyRAD')
        self.logger.setLevel(logging.INFO)
        
        # Clear all existing handlers to re-instantiate the logger
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        self.log_file       = log_file
        self.log_dir        = log_dir
        self.flush_file     = log_file is not None
        # Backward compatibility: if prefix_date is set (legacy), use it, else use prefix_time
        if prefix_date is not None:
            warnings.warn(
                "The 'prefix_date' argument is deprecated and will be removed by 2025 Aug."
                "Please use 'prefix_time' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.prefix_time = prefix_date
        else:
            self.prefix_time = prefix_time
        self.prefix_jobid   = prefix_jobid
        self.append_to_file = append_to_file
        self.show_timestamp = show_timestamp

        # Create console handler
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s' if show_timestamp else '%(message)s')
        self.console_handler.setFormatter(formatter)
        
        # Create a buffer for file logs
        self.log_buffer = io.StringIO()
        self.buffer_handler = logging.StreamHandler(self.log_buffer)
        self.buffer_handler.setLevel(logging.INFO)
        self.buffer_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(self.console_handler)
        self.logger.addHandler(self.buffer_handler)
        
        # Print logger information
        vprint("### PtyRAD Logger configuration ###")
        vprint(f"log_file       = '{self.log_file}'. If log_file = None, no log file will be created.")
        vprint(f"log_dir        = '{self.log_dir}'. If log_dir = 'auto', then log will be saved to `output_path` or 'logs/'.")
        vprint(f"flush_file     = {self.flush_file}. Automatically set to True if `log_file is not None`")
        vprint(f"prefix_time    = {self.prefix_time}. If true, preset strings ('date', 'time', 'datetime'), or a string of time format, a datetime str is prefixed to the `log_file`.")
        vprint(f"prefix_jobid   = '{self.prefix_jobid}'. If not 0, it'll be prefixed to the log file. This is used for hypertune mode with multiple GPUs.")
        vprint(f"append_to_file = {self.append_to_file}. If true, logs will be appended to the existing file. If false, the log file will be overwritten.")
        vprint(f"show_timestamp = {self.show_timestamp}. If true, the printed information will contain a timestamp.")
        vprint(' ')

    def flush_to_file(self, log_dir=None, append_to_file=None):
        """
        Flushes buffered logs to a file based on user-defined file mode (append or write)
        """
        
        # Set log_dir
        if log_dir is None:
            if self.log_dir == 'auto':
                log_dir = 'logs'
            else:
                log_dir = self.log_dir

        # Set file_mode
        if append_to_file is None:
            append_to_file = self.append_to_file
        file_mode = 'a' if append_to_file else 'w'
        
        # Set file name
        log_file = self.log_file
        if self.prefix_jobid != 0:
            log_file = str(self.prefix_jobid).zfill(2) + '_' + log_file
        
        # Set prefix_time
        prefix_time = self.prefix_time
        if prefix_time is True or (isinstance(prefix_time, str) and prefix_time):
            time_str = get_time(prefix_time)
            log_file = f"{time_str}_{log_file}"
        
        show_timestamp = self.show_timestamp
        
        if self.flush_file:
            # Ensure the log directory exists
            os.makedirs(log_dir, exist_ok=True)
            log_file_path = os.path.join(log_dir, log_file)

            # Write the buffered logs to the specified file
            try:
                with open(log_file_path, file_mode, encoding="utf-8") as f:
                    f.write(self.log_buffer.getvalue())
            except UnicodeEncodeError as e:
                vprint(f"[WARNING] Failed to write log due to Unicode issue: {e}")
                with open(log_file_path, file_mode, encoding="ascii", errors="replace") as f:
                    f.write(self.log_buffer.getvalue())

            # Clear the buffer
            self.log_buffer.truncate(0)
            self.log_buffer.seek(0)

            # Set up a file handler for future logging to the file
            self.file_handler = logging.FileHandler(log_file_path, mode='a')  # Always append after initial flush
            self.file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s' if show_timestamp else '%(message)s'))
            self.logger.addHandler(self.file_handler)
            vprint(f"### Log file is flushed (created) as {log_file_path} ###")
        else:
            self.file_handler = None
            vprint(f"### Log file is not flushed (created) because log_file is set to {self.log_file} ###")
        vprint(' ')
        
    def close(self):
        """Closes the file handler if it exists."""
        if self.file_handler is not None:
            self.file_handler.flush()
            self.file_handler.close()
            self.logger.removeHandler(self.file_handler)
            self.file_handler = None

def print_system_info():
    import os
    import platform
    import sys
    
    vprint("### System information ###")
    
    # Operating system information
    vprint(f"Operating System: {platform.system()} {platform.release()}")
    vprint(f"OS Version: {platform.version()}")
    vprint(f"Machine: {platform.machine()}")
    vprint(f"Processor: {platform.processor()}")
    
    # CPU cores
    if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
        cpus =  int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    else:
        # Fallback to the total number of CPU cores on the node
        cpus = os.cpu_count()
    vprint(f"Available CPU cores: {cpus}")
    
    # Memory information
    if 'SLURM_MEM_PER_NODE' in os.environ:
        # Memory allocated per node by SLURM (in MB)
        mem_total = int(os.environ['SLURM_MEM_PER_NODE']) / 1024  # Convert MB to GB
        vprint(f"SLURM-Allocated Total Memory: {mem_total:.2f} GB")
    elif 'SLURM_MEM_PER_CPU' in os.environ:
        # Memory allocated per CPU by SLURM (in MB)
        mem_total = int(os.environ['SLURM_MEM_PER_CPU']) * cpus / 1024  # Convert MB to GB
        vprint(f"SLURM-Allocated Total Memory: {mem_total:.2f} GB")
    else:
        try:
            import psutil
            # Fallback to system memory information
            mem = psutil.virtual_memory()
            vprint(f"Total Memory: {mem.total / (1024 ** 3):.2f} GB")
            vprint(f"Available Memory: {mem.available / (1024 ** 3):.2f} GB")
        except ImportError:
            vprint("Memory information will be available after `conda install conda-forge::psutil`")
    vprint(" ")
            
    # GPU information
    print_gpu_info()
    vprint(" ")
    
    # Python version and executable
    vprint("### Python information ###")
    vprint(f"Python Executable: {sys.executable}")
    vprint(f"Python Version: {sys.version}")
    vprint(" ")
    
    # Packages information (numpy, PyTorch, Optuna, Accelerate, PtyRAD)
    print_packages_info()
    vprint(" ")

def print_gpu_info():
    vprint("### GPU information ###")
    try:
        import torch
        if torch.backends.cuda.is_built() and torch.cuda.is_available():
            vprint(f"CUDA Available: {torch.cuda.is_available()}")
            vprint(f"CUDA Version: {torch.version.cuda}")
            vprint(f"Available CUDA GPUs: {[torch.cuda.get_device_name(d) for d in range(torch.cuda.device_count())]}")
            vprint(f"MIG (Multi-Instance GPU) mode = {is_mig_enabled()}")
            vprint("INFO: MIG splits a physical GPU into multiple GPU slices, but multiGPU does not support these MIG slices.")
            vprint("      -> If you're doing normal reconstruction/hypertune, you can safely ignore this.")
            vprint("      -> If you want to do multiGPU, you must provide multiple 'full' GPUs that are not in MIG mode.")
        elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
            vprint(f"MPS Available: {torch.backends.mps.is_available()}")
        elif torch.backends.cuda.is_built() or torch.backends.mps.is_built():
            vprint("WARNING: GPU support built with PyTorch, but could not find any existing / compatible GPU device.")
            vprint("         PtyRAD will fall back to CPU which is much slower in performance")
            vprint("         -> If you're using a CPU-only machine, you can safely ignore this.")
            vprint("         -> If you believe you *do* have a GPU, please check the compatibility:")
            vprint("           - Are the correct NVIDIA drivers installed?")
            vprint("           - Is your CUDA runtime version compatible with PyTorch?")
            vprint("           Tips: Run `nvidia-smi` in your terminal for NVIDIA driver and CUDA runtime information.")
            vprint("           Tips: Run `conda list torch` in your terminal (with `ptyrad` environment activated) to check the installed PyTorch version.")
        else:
            vprint("WARNING: No GPU backend (CUDA or MPS) built into this PyTorch install.")
            vprint("         PtyRAD will fall back to CPU which is much slower in performance")
            vprint("         Please consider reinstalling PyTorch with GPU support if available.")
            vprint("         See https://github.com/chiahao3/ptyrad for PtyRAD installation guide.")
    except ImportError:
        vprint("WARNING: No GPU information because PyTorch can't be imported.")
        vprint("         Please install PyTorch because it's the crucial dependency of PtyRAD.")
        vprint("         See https://github.com/chiahao3/ptyrad for PtyRAD installation guide.")
    
def print_packages_info():
    import importlib
    import importlib.metadata
    vprint("### Packages information ###")
    
    # Print package versions
    packages = [
        ("Numpy", "numpy"),
        ("PyTorch", "torch"),
        ("Optuna", "optuna"),
        ("Accelerate", "accelerate"),
    ]

    # Check versions for relevant packages
    for display_name, module_name in packages:
        try:
            # Try to get the version from package metadata (installed version)
            version = importlib.metadata.version(module_name)
            vprint(f"{display_name} Version (metadata): {version}")
        except importlib.metadata.PackageNotFoundError:
            vprint(f"{display_name} not found in the environment.")
        except Exception as e:
            vprint(f"Error retrieving version for {display_name}: {e}")
    
    # Check the version and path of the used PtyRAD package
    # In general:
    # - `ptyrad.__version__` reflects the actual code you're running (from source files).
    # - `importlib.metadata.version("ptyrad")` reflects the version during install.
    # 
    # Note that we're focusing on the version/path of the actual imported PtyRAD.
    # If there are both an installed version of PtyRAD in the environment (site-packages/) and a local copy in the working directory (src/ptyrad),
    # Python will prioritize the version in the working directory.
    #
    # When using `pip install -e .`, only the version metadata gets recorded, which won't be updated until you reinstall.
    # As a result, a user who pulls new code from the repo will have their `__init__.py` updated, but the version metadata recorded by pip will remain unchanged.
    # Therefore, it is better to retrieve the version directly from `module.__version__` for now, as this will reflect the actual local version being used.
    # In a release install (pip or conda), metadata and __version__ will match due to the dynamic version in pyproject.toml
    # During editable installs, metadata may lag behind source changes.
    try:
        # Import ptyrad (which will prioritize the local version if available)
        module = importlib.import_module('ptyrad')
        runtime_version = module.__version__
        metadata_version = importlib.metadata.version("ptyrad")
        vprint(f"PtyRAD Version (ptyrad/__init__.py): {runtime_version}")
        vprint(f"PtyRAD is located at: {module.__file__}") # For editable install this will be in package src/, while full install would make a copy at site-packages/
        
        if runtime_version and metadata_version and runtime_version != metadata_version:
            vprint("WARNING: Version mismatch detected!")
            vprint(f"  Runtime version : {runtime_version} (retrieved from current source file: ptyrad/__init__.py)")
            vprint(f"  Metadata version: {metadata_version} (recorded during previous `pip/conda install`)")
            vprint("  This likely means you downloaded new codes from repo but forgot to update the installed metadata.")
            vprint("  This does not affect the code execution because the runtime version of code is always used, but this can lead to misleading version logs.")
            vprint("  To fix this, re-run: `pip install -e . --no-deps` at the package root directory.")
        
    except ImportError:
        vprint("PtyRAD not found locally")
    except AttributeError:
        vprint("PtyRAD imported, but no __version__ attribute found.")
    except Exception as e:
        vprint(f"Error retrieving version for PtyRAD: {e}")

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

@torch.compiler.disable
def vprint(*args, verbose=True, **kwargs):
    """Verbose print/logging with individual control, only for rank 0 in DDP."""
    if verbose and (not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0):
        logger = logging.getLogger('PtyRAD')
        if logger.hasHandlers():
            logger.info(' '.join(map(str, args)), **kwargs)
        else:
            print(*args, **kwargs)

def vprint_nested_dict(d, indent=0, verbose=True, leaf_inline_threshold=6):
    indent_str = "    " * indent
    for key, value in d.items():
        if isinstance(value, dict):
            # Check if this is a flat leaf dict
            is_flat_leaf = all(not isinstance(v, (dict, list)) for v in value.values())
            if is_flat_leaf and len(value) <= leaf_inline_threshold:  # Determine whether to print inline or not
                flat = ", ".join(f"{k}: {repr(v)}" for k, v in value.items())
                vprint(f"{indent_str}{key}: {{{flat}}}", verbose=verbose)
            else:
                vprint(f"{indent_str}{key}:", verbose=verbose)
                vprint_nested_dict(value, indent + 1, verbose=verbose)
        elif isinstance(value, list) and all(not isinstance(i, (dict, list)) for i in value):
            vprint(f"{indent_str}{key}: {value}", verbose=verbose)
        else:
            vprint(f"{indent_str}{key}: {repr(value)}", verbose=verbose)

def expand_presets(input_list, presets):
    expanded = []
    for tag in input_list:
        if tag in presets:
            expanded.extend(presets[tag])
        else:
            expanded.append(tag)
    return list(dict.fromkeys(expanded))  # Removes duplicates, keeps order

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
    
def get_time(time_format: Union[bool, str, None] = 'date') -> str:
    """
    Returns a formatted timestamp string based on `time_format`.

    Args:
        time_format (bool or str): Controls the time formatting behavior.
            - True: Use default date format ("%Y%m%d").
            - False, None, or "": Disable timestamp and return an empty string.
            - "date": Use date format ("%Y%m%d").
            - "datetime": Use date and time format ("%Y%m%d_%H%M%S").
            - "time": Use time-only format ("%H%M%S").
            - Custom strftime format (e.g., "%Y-%m-%d %H:%M") is also supported.

    Returns:
        str: Formatted timestamp string, or an empty string if disabled.
    """
    from datetime import date, datetime

    if not time_format:
        return ""

    if isinstance(time_format, bool):
        fmt = "%Y%m%d"
    elif isinstance(time_format, str):
        presets = {
            "date": "%Y%m%d",
            "datetime": "%Y%m%d_%H%M%S",
            "time": "%H%M%S",
        }
        fmt = presets.get(time_format.lower(), time_format)
    else:
        raise TypeError(f"time_format must be a bool or str, got {type(time_format).__name__}")

    # Choose datetime vs date object depending on format
    try:
        if any(tok in fmt for tok in ("%H", "%M", "%S")):
            return datetime.now().strftime(fmt)
        else:
            return date.today().strftime(fmt)
    except ValueError as e:
        raise ValueError(f"Invalid time format string: {fmt!r}") from e

@torch.compiler.disable
def time_sync():
    # PyTorch doesn't have a direct exposed API to check the selected default device 
    # so we'll be checking these .is_available() just to prevent error.
    # Luckily these checks won't really affect the performance.
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Check if MPS (Metal Performance Shaders) is available (macOS only)
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()
    
    # Measure the time
    t = perf_counter()
    return t

def parse_sec_to_time_str(seconds):
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if days > 0:
        return f"{int(days)} day {int(hours)} hr {int(minutes)} min {secs:.3f} sec"
    elif hours > 0:
        return f"{int(hours)} hr {int(minutes)} min {secs:.3f} sec"
    elif minutes > 0:
        return f"{int(minutes)} min {secs:.3f} sec"
    else:
        return f"{secs:.3f} sec"

def parse_hypertune_params_to_str(hypertune_params):
    
    hypertune_str = ''
    for key, value in hypertune_params.items():
        if key[-2:].lower() == "lr":
            hypertune_str += f"_{key}_{value:.1e}"
        elif isinstance(value, (int, float)):
            hypertune_str += f"_{key}_{value:.3g}"
        else:
            hypertune_str += f"_{key}_{value}"
    
    return hypertune_str

def safe_filename(filepath, verbose=False):
    """
    Ensures a filepath is safe across platforms by:
    1. Converting relative paths to absolute
    2. Limiting individual components to 255 characters
    3. Handling total path length restrictions
    4. Providing feedback when corrections are made
    
    Args:
        filepath: The original filepath to make safe
        verbose: Whether to print messages about corrections (default: False)
    
    Returns:
        A modified filepath that should work across platforms
    """
    # Store original path for reporting
    original_path = filepath
    
    # Handle relative paths by converting to absolute
    filepath = os.path.abspath(filepath)
    
    # Platform detection
    is_windows = platform.system() == 'Windows'
    
    # Check if path already has long path prefix on Windows
    has_long_prefix = is_windows and filepath.startswith("\\\\?\\")
    
    # Early return if path is already valid
    if not has_long_prefix:
        # Check individual component limit (255 chars)
        components_valid = True
        sep = '\\' if is_windows else '/'
        parts = filepath.split(sep)
        for part in parts:
            if len(part) > 255:
                components_valid = False
                break
        
        # Check total path length limit
        length_valid = (len(filepath) <= 260) if is_windows else True
        
        # If everything is valid, return the absolute path
        if components_valid and length_valid:
            return filepath
    
    # Path requires correction - continue with fixing logic
    # Path separator based on platform
    sep = '\\' if is_windows else '/'
    
    # Split path into directory and filename
    directory, filename = os.path.split(filepath)
    
    # Track if any changes were made
    changes_made = False
    
    # Limit filename component to 255 chars (preserve extension)
    if len(filename) > 255:
        changes_made = True
        name, ext = os.path.splitext(filename)
        max_name_length = 255 - len(ext)
        filename = name[:max_name_length] + ext
    
    # Handle directory components (limit each to 255 chars)
    if directory:
        parts = directory.split(sep)
        for i, part in enumerate(parts):
            if len(part) > 255:
                changes_made = True
                parts[i] = part[:255]
        directory = sep.join(parts)
    
    # Recombine path
    result_path = os.path.join(directory, filename)
    
    # Handle Windows total path length
    if is_windows and len(result_path) > 260:
        changes_made = True
        # If still too long, apply the \\?\ prefix for long path support
        if not result_path.startswith("\\\\?\\"):
            # Ensure we're working with an absolute path for the \\?\ prefix
            result_path = "\\\\?\\" + os.path.abspath(result_path)
    
    # Provide feedback if corrections were made
    if changes_made and verbose:
        print("Path corrected for compatibility:")
        print(f"  Original: {original_path}")
        print(f"  Corrected: {result_path}")
    
    return result_path

def handle_hdf5_types(x):
    """
    Convert data to native Python or NumPy types. Especially when loaded by h5py.

    Handles special cases like MATLAB v7.3 complex128 data types and ensures
    that data is converted to a format compatible with native Python or NumPy.

    Also handles sentinel string "__NONE__" as a substitute for None in HDF5.

    Args:
        x: The input data to be converted.

    Returns:
        The converted data into native Python or NumPy types.
    """
    # Handle scalar Numpy types
    if isinstance(x, np.generic):
        x = x.item()

    # Handle 0-dimensional Numpy arrays (convert to Python scalars) as they were probably forced by HDF5
    if isinstance(x, np.ndarray) and x.ndim == 0:
        x = x.item()

    # Handle bytes (e.g., HDF5 strings or sentinel)
    if isinstance(x, bytes):
        try:
            x = x.decode('utf-8')
        except UnicodeDecodeError:
            return x  # Leave undecodable bytes unchanged

    # Convert sentinel string to None — only safe for scalar strings
    if isinstance(x, str) and x == "__NONE__":
        return None

    # Handle MATLAB-style complex128 compound dtype
    if isinstance(x, np.ndarray) and x.dtype == [('real', '<f8'), ('imag', '<f8')]:
        vprint(f"Detected data.shape = {x.shape} with data.dtype = {x.dtype}. Casting back to 'complex128'.")
        return x.view(np.complex128)

    # Convert 1D array of strings (or object-dtype strings) to Python list of str
    if isinstance(x, np.ndarray) and x.ndim == 1:
        if np.issubdtype(x.dtype, np.str_) or np.issubdtype(x.dtype, np.object_):
            try:
                return [i.decode('utf-8') if isinstance(i, bytes) else str(i) for i in x]
            except Exception:
                pass  # fallback to returning as-is
            
    # Try parsing stringified literals
    if isinstance(x, str):
        import ast
        try:
            parsed = ast.literal_eval(x)
            return parsed
        except (ValueError, SyntaxError):
            pass
    
    return x

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

def tensors_to_ndarrays(data):
    """
    Recursively convert all torch.Tensor instances in any nested structure 
    (including lists, dicts, and tuples) into numpy.ndarray.

    This function supports both single objects and arbitrarily nested 
    container types. Non-tensor types are returned unchanged.

    Args:
        data: Input data which can be nested dict, list, tuple, torch.Tensor, or other.

    Returns:
        The same structure with torch.Tensor replaced by numpy.ndarray.
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, dict):
        return {k: tensors_to_ndarrays(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [tensors_to_ndarrays(x) for x in data]
    elif isinstance(data, tuple):
        return tuple(tensors_to_ndarrays(x) for x in data)
    else:
        return data
    
def ndarrays_to_tensors(data, device='cuda'):
    """
    Recursively convert all numpy.ndarray instances in any nested structure 
    (including lists, dicts, and tuples) into torch.Tensor on the specified device.

    This function supports both single objects and arbitrarily nested 
    container types. Non-array types are returned unchanged.

    Args:
        data: Input data which can be nested dict, list, tuple, np.ndarray, or other.
        device (str): Device on which to place the tensors. Default is 'cuda'.

    Returns:
        The same structure with numpy.ndarray replaced by torch.Tensor.
    """
    if isinstance(data, np.ndarray):
        return torch.tensor(data).to(device)
    elif isinstance(data, dict):
        return {k: ndarrays_to_tensors(v, device=device) for k, v in data.items()}
    elif isinstance(data, list):
        return [ndarrays_to_tensors(x, device=device) for x in data]
    elif isinstance(data, tuple):
        return tuple(ndarrays_to_tensors(x, device=device) for x in data)
    else:
        return data