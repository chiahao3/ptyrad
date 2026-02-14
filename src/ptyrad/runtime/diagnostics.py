import logging
import subprocess

from ptyrad.runtime.logging import report

logger = logging.getLogger(__name__)


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

def print_system_info():
    import os
    import platform
    import sys
    
    report("### System information ###")
    
    # Operating system information
    report(f"Platform: {platform.platform()}")
    report(f"Operating System: {platform.system()} {platform.release()}")
    report(f"OS Version: {platform.version()}")
    report(f"Machine: {platform.machine()}")
    report(f"Processor: {platform.processor()}")
    
    # CPU cores
    if 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
        cpus =  int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
    else:
        # Fallback to the total number of CPU cores on the node
        cpus = os.cpu_count()
    report(f"Available CPU cores: {cpus}")
    
    # Memory information
    if 'SLURM_MEM_PER_NODE' in os.environ:
        # Memory allocated per node by SLURM (in MB)
        mem_total = int(os.environ['SLURM_MEM_PER_NODE']) / 1024  # Convert MB to GB
        report(f"SLURM-Allocated Total Memory: {mem_total:.2f} GB")
    elif 'SLURM_MEM_PER_CPU' in os.environ:
        # Memory allocated per CPU by SLURM (in MB)
        mem_total = int(os.environ['SLURM_MEM_PER_CPU']) * cpus / 1024  # Convert MB to GB
        report(f"SLURM-Allocated Total Memory: {mem_total:.2f} GB")
    else:
        try:
            import psutil
            # Fallback to system memory information
            mem = psutil.virtual_memory()
            report(f"Total Memory: {mem.total / (1024 ** 3):.2f} GB")
            report(f"Available Memory: {mem.available / (1024 ** 3):.2f} GB")
        except ImportError:
            report("Memory information will be available after `conda install conda-forge::psutil`")
    report(" ")
            
    # GPU information
    print_gpu_info()
    report(" ")
    
    # Python version and executable
    report("### Python information ###")
    report(f"Python Executable: {sys.executable}")
    report(f"Python Version: {sys.version}")
    report(" ")
    
    # Packages information (numpy, PyTorch, Optuna, Accelerate, PtyRAD)
    print_packages_info()
    report(" ")

def print_gpu_info():
    report("### GPU information ###")
    try:
        import torch
        if torch.backends.cuda.is_built() and torch.cuda.is_available():
            report(f"CUDA Available: {torch.cuda.is_available()}")
            report(f"CUDA Version: {torch.version.cuda}")
            report(f"Available CUDA GPUs: {[torch.cuda.get_device_name(d) for d in range(torch.cuda.device_count())]}")
            report(f"CUDA Compute Capability: {[f'{major}.{minor}' for (major, minor) in [torch.cuda.get_device_capability(d) for d in range(torch.cuda.device_count())]]}")
            report("  INFO: For torch.compile with Triton, you'll need CUDA GPU with Compute Capability >= 7.0.")
            report("        In addition, Triton does not directly support Windows.")
            report("        For Windows users, please follow the instruction and download `triton-windows` from https://github.com/woct0rdho/triton-windows.")
            report(f"MIG (Multi-Instance GPU) mode = {is_mig_enabled()}")
            report("  INFO: MIG splits a physical GPU into multiple GPU slices, but multiGPU does not support these MIG slices.")
            report("        In addition, multiGPU is currently only available on Linux due to the limited NCCL support.")
            report("      -> If you're doing normal reconstruction/hypertune, you can safely ignore this.")
            report("      -> If you want to do multiGPU, you must provide multiple 'full' GPUs that are not in MIG mode.")
        elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
            report(f"MPS Available: {torch.backends.mps.is_available()}")
        elif torch.backends.cuda.is_built() or torch.backends.mps.is_built():
            report("WARNING: GPU support built with PyTorch, but could not find any existing / compatible GPU device.")
            report("         PtyRAD will fall back to CPU which is much slower in performance")
            report("         -> If you're using a CPU-only machine, you can safely ignore this.")
            report("         -> If you believe you *do* have a GPU, please check the compatibility:")
            report("           - Are the correct NVIDIA drivers installed?")
            report("           - Is your CUDA runtime version compatible with PyTorch?")
            report("           Tips: Run `nvidia-smi` in your terminal for NVIDIA driver and CUDA runtime information.")
            report("           Tips: Run `conda list torch` in your terminal (with `ptyrad` environment activated) to check the installed PyTorch version.")
        else:
            report("WARNING: No GPU backend (CUDA or MPS) built into this PyTorch install.")
            report("         PtyRAD will fall back to CPU which is much slower in performance")
            report("         Please consider reinstalling PyTorch with GPU support if available.")
            report("         See https://github.com/chiahao3/ptyrad for PtyRAD installation guide.")
    except ImportError:
        report("WARNING: No GPU information because PyTorch can't be imported.")
        report("         Please install PyTorch because it's the crucial dependency of PtyRAD.")
        report("         See https://github.com/chiahao3/ptyrad for PtyRAD installation guide.")
    
def print_packages_info():
    import importlib
    import importlib.metadata
    report("### Packages information ###")
    
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
            report(f"{display_name} Version (metadata): {version}")
        except importlib.metadata.PackageNotFoundError:
            report(f"{display_name} not found in the environment.")
        except Exception as e:
            report(f"Error retrieving version for {display_name}: {e}")
    
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
        report(f"PtyRAD Version (ptyrad/__init__.py): {runtime_version}")
        report(f"PtyRAD is located at: {module.__file__}") # For editable install this will be in package src/, while full install would make a copy at site-packages/
        
        if runtime_version and metadata_version and runtime_version != metadata_version:
            report("WARNING: Version mismatch detected!")
            report(f"  Runtime version : {runtime_version} (retrieved from current source file: ptyrad/__init__.py)")
            report(f"  Metadata version: {metadata_version} (recorded during previous `pip/conda install`)")
            report("  This likely means you downloaded new codes from repo but forgot to update the installed metadata.")
            report("  This does not affect the code execution because the runtime version of code is always used, but this can lead to misleading version logs.")
            report("  To fix this, re-run: `pip install -e . --no-deps` at the package root directory.")
        
    except ImportError:
        report("PtyRAD not found locally")
    except AttributeError:
        report("PtyRAD imported, but no __version__ attribute found.")
    except Exception as e:
        report(f"Error retrieving version for PtyRAD: {e}")