import io
import logging
import os
import subprocess
import warnings
from typing import Union

import torch
import torch.distributed as dist

from .timing import get_time


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

def print_system_info():
    import os
    import platform
    import sys
    
    vprint("### System information ###")
    
    # Operating system information
    vprint(f"Platform: {platform.platform()}")
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
            vprint(f"CUDA Compute Capability: {[f'{major}.{minor}' for (major, minor) in [torch.cuda.get_device_capability(d) for d in range(torch.cuda.device_count())]]}")
            vprint("  INFO: For torch.compile with Triton, you'll need CUDA GPU with Compute Capability >= 7.0.")
            vprint("        In addition, Triton does not directly support Windows.")
            vprint("        For Windows users, please follow the instruction and download `triton-windows` from https://github.com/woct0rdho/triton-windows.")
            vprint(f"MIG (Multi-Instance GPU) mode = {is_mig_enabled()}")
            vprint("  INFO: MIG splits a physical GPU into multiple GPU slices, but multiGPU does not support these MIG slices.")
            vprint("        In addition, multiGPU is currently only available on Linux due to the limited NCCL support.")
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