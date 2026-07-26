"""
JIT (``torch.compile``) capability auto-detection with graceful fallback.

PtyRAD can run its loss/optimizer steps through PyTorch's JIT compiler for a
1.3-1.9x speedup, but whether that actually works depends on the machine:
the OS, the Python version, the PyTorch build, the accelerator (CUDA / MPS /
CPU), the GPU compute capability, and whether Triton (GPU) or a C++ compiler
(CPU) is installed. Rather than asking users to figure that out, the default
``compiler_configs: {'enable': 'auto'}`` lets PtyRAD detect it right before
the reconstruction loop starts and silently fall back to eager mode when JIT
is not achievable.

Detection is done in two stages:

1. A cheap *static* check of the environment (PyTorch version, Dynamo support,
   backend availability, GPU compute capability, Triton / C++ compiler).
2. A *functional* probe that actually compiles and runs a tiny function
   (forward + backward) on the target device, which is the only reliable way
   to catch broken toolchains that look fine on paper.

Both stages are cached per-process, so repeated calls (e.g. one per Optuna
trial during hypertune) pay the cost only once.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import platform
import shutil
import warnings
from functools import lru_cache
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# torch.compile was introduced in PyTorch 2.0
MIN_TORCH_VERSION = (2, 0)

# Triton (the GPU backend used by TorchInductor) requires CUDA compute capability >= 7.0 (Volta)
MIN_CUDA_CAPABILITY = (7, 0)

# TorchInductor gained a Metal (MPS) codegen backend in PyTorch 2.7,
# earlier versions raise or silently degrade when compiling on Apple Silicon GPUs
MIN_TORCH_VERSION_MPS = (2, 7)

# Device types whose TorchInductor codegen path goes through Triton
TRITON_DEVICE_TYPES = ("cuda", "xpu", "hpu")

TRITON_WINDOWS_HINT = (
    "Triton does not officially support Windows. "
    "Install `triton-windows` (https://github.com/woct0rdho/triton-windows) to enable JIT on Windows."
)


def _torch_version() -> Tuple[int, int]:
    """Return the (major, minor) version of the installed PyTorch as a tuple of ints."""
    import torch

    try:
        return tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2])
    except ValueError:
        # Nightly / custom builds with non-numeric version strings, assume they're recent enough
        return (99, 99)


def resolve_device(device=None) -> str:
    """Resolve the canonical device string ('cuda:1', 'mps', 'cpu') used for JIT detection.

    The GPU index is preserved (and filled in from the current CUDA device when the caller
    only gave a bare 'cuda') so that the capability check and the functional probe both
    target the exact GPU the reconstruction will run on, which matters on heterogeneous
    hosts where the visible GPUs have different compute capabilities.

    Args:
        device (torch.device or str or None, optional): The target device. When None
            (e.g. multi-GPU runs where `accelerate` owns device placement), it is
            inferred from the available accelerators.

    Returns:
        str: The resolved device string, suitable for `torch.device()`.
    """
    import torch

    if device is not None:
        dev = torch.device(device)
    elif torch.cuda.is_available():
        dev = torch.device("cuda")
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    # Pin down which CUDA GPU is meant, since that's the one PyTorch would allocate on
    if dev.type == "cuda" and dev.index is None and torch.cuda.is_available():
        try:
            dev = torch.device("cuda", torch.cuda.current_device())
        except Exception:
            pass

    return str(dev)


def _has_triton() -> bool:
    """Check whether a usable Triton installation is present."""
    try:
        # Available since PyTorch 2.2, this also validates the device backend, not just the import
        from torch.utils._triton import has_triton

        return bool(has_triton())
    except Exception:
        # Fall back to a plain import check on older PyTorch
        try:
            return importlib.util.find_spec("triton") is not None
        except Exception:
            return False


def _has_cxx_compiler() -> bool:
    """Check whether a C++ compiler is on PATH (needed by TorchInductor's CPU backend)."""
    if platform.system() == "Windows":
        candidates = ("cl", "clang-cl", "g++")
    else:
        candidates = (os.environ.get("CXX"), os.environ.get("CC"), "g++", "clang++", "c++", "gcc")
    return any(shutil.which(c) for c in candidates if c)


def _cuda_capability(device_str: str) -> Optional[Tuple[int, int]]:
    """Return the CUDA compute capability of the selected GPU, or None if it's not a CUDA device.

    When the device carries no index (so the selected GPU is unknown), the lowest capability
    among the visible GPUs is returned to stay conservative.
    """
    import torch

    dev = torch.device(device_str)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        if dev.index is not None:
            return torch.cuda.get_device_capability(dev.index)
        caps = [torch.cuda.get_device_capability(d) for d in range(torch.cuda.device_count())]
        return min(caps) if caps else None
    except Exception:
        return None


@lru_cache(maxsize=None)
def check_jit_support(device_str: str = "cpu", backend: str = "inductor") -> Tuple[bool, str]:
    """Statically check whether ``torch.compile`` is expected to work on this machine.

    This is a cheap, side-effect-free inspection of the environment. It never
    raises: any unexpected failure is reported as "unsupported" with a reason.

    Args:
        device_str (str, optional): Target device, either a bare type ('cuda', 'mps', 'cpu')
            or an indexed device ('cuda:1') so that per-GPU properties are read from the GPU
            actually selected for the reconstruction. Defaults to 'cpu'.
        backend (str, optional): The ``torch.compile`` backend. Defaults to 'inductor'.

    Returns:
        tuple[bool, str]: (supported, reason) where `reason` explains the verdict and,
        when unsupported, how the user could enable JIT on their machine.
    """
    try:
        import torch
    except ImportError:
        return False, "PyTorch is not importable"

    system = platform.system()
    device_type = torch.device(device_str).type

    # (1) torch.compile must exist at all
    version = _torch_version()
    if not hasattr(torch, "compile") or version < MIN_TORCH_VERSION:
        return False, (
            f"PyTorch {torch.__version__} has no usable torch.compile "
            f"(requires >= {MIN_TORCH_VERSION[0]}.{MIN_TORCH_VERSION[1]})"
        )

    # (2) TorchDynamo must support this Python version / platform combination
    try:
        from torch._dynamo import is_dynamo_supported

        if not is_dynamo_supported():
            return False, (
                f"TorchDynamo is unsupported for Python {platform.python_version()} "
                f"with PyTorch {torch.__version__}"
            )
    except ImportError:
        pass  # Too old or trimmed-down build, let the functional probe be the judge

    # (3) Backend-specific requirements
    if backend == "inductor":
        try:
            from torch._dynamo import is_inductor_supported

            if not is_inductor_supported():
                return False, f"TorchInductor is unsupported by PyTorch {torch.__version__} on this platform"
        except ImportError:
            pass

        if device_type in TRITON_DEVICE_TYPES:
            if not _has_triton():
                hint = f" {TRITON_WINDOWS_HINT}" if system == "Windows" else ""
                return False, f"Triton is not available for the '{device_type}' TorchInductor backend.{hint}"

            capability = _cuda_capability(device_str)
            if capability is not None and capability < MIN_CUDA_CAPABILITY:
                return False, (
                    f"CUDA compute capability {capability[0]}.{capability[1]} of device '{device_str}' "
                    f"is below the {MIN_CUDA_CAPABILITY[0]}.{MIN_CUDA_CAPABILITY[1]} required by Triton"
                )

        elif device_type == "mps":
            if version < MIN_TORCH_VERSION_MPS:
                return False, (
                    f"TorchInductor has no Metal (MPS) backend in PyTorch {torch.__version__} "
                    f"(requires >= {MIN_TORCH_VERSION_MPS[0]}.{MIN_TORCH_VERSION_MPS[1]})"
                )

        elif device_type == "cpu":
            if not _has_cxx_compiler():
                return False, (
                    "No C++ compiler found on PATH, which TorchInductor needs to build CPU kernels "
                    "(install g++/clang++ on Linux/macOS, or MSVC on Windows)"
                )

    return True, (
        f"PyTorch {torch.__version__} on {system} with device '{device_str}' "
        f"and backend '{backend}' meets the JIT requirements"
    )


def _jit_probe_fn(x):
    """Tiny function used to smoke-test the JIT compiler (kept at module level so Dynamo can trace it)."""
    return (x * x + 1.0).sum()


def _sync_device(device) -> None:
    """Synchronize the device so asynchronous kernel failures surface inside the probe."""
    import torch

    try:
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        elif device.type == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize() # As of pytorch 2.10, torch.mps.synchronize doesn't take any arg
    except Exception:
        pass


@lru_cache(maxsize=None)
def probe_jit_compile(
    device_str: str = "cpu",
    backend: str = "inductor",
    fullgraph: bool = False,
    dynamic: Optional[bool] = None,
) -> Tuple[bool, str]:
    """Functionally verify ``torch.compile`` by compiling and running a tiny function.

    This actually exercises the whole toolchain (Dynamo trace, backend codegen,
    kernel build, forward, and backward) on the target device, which is the only
    reliable way to catch environments that pass the static check but have a
    broken compiler, missing CUDA headers, or an incompatible Triton build.

    The probe compiles a 16x16 element function, so the cost is dominated by the
    one-time backend warmup (a few seconds at most) and is cached per-process.

    Note:
        A hard crash (segfault) inside a broken backend cannot be caught here,
        only Python-level exceptions are handled.

    Args:
        device_str (str, optional): Target device, e.g. 'cpu', 'mps', or 'cuda:1'. The probe runs
            on the exact device given so it reflects the GPU used by the reconstruction.
            Defaults to 'cpu'.
        backend (str, optional): The ``torch.compile`` backend. Defaults to 'inductor'.
        fullgraph (bool, optional): Whether to require a graph break-free trace. Defaults to False.
        dynamic (bool or None, optional): Dynamic shape handling. Defaults to None.

    Returns:
        tuple[bool, str]: (works, reason) describing whether the probe compiled and ran cleanly.
    """
    import torch

    try:
        device = torch.device(device_str)
    except Exception as err:
        return False, f"invalid device type '{device_str}' ({err})"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            x = torch.randn(16, 16, device=device, requires_grad=True)
            compiled_fn = torch.compile(_jit_probe_fn, backend=backend, fullgraph=fullgraph, dynamic=dynamic)
            out = compiled_fn(x)
            out.backward()
            _sync_device(device)

            # Guard against a backend that "succeeds" but returns garbage
            if x.grad is None or not bool(torch.isfinite(out).all()) or not bool(torch.isfinite(x.grad).all()):
                return False, "the compiled probe function returned non-finite values"

    except Exception as err:
        # torch.compile failures surface as BackendCompilerFailed, Unsupported, ImportError, OSError, RuntimeError...
        return False, f"torch.compile probe failed with {type(err).__name__}: {err}"
    finally:
        # Leave no compiled state behind, the reconstruction loop resets and compiles its own graphs
        try:
            torch._dynamo.reset()
        except Exception:
            pass

    return True, f"torch.compile probe succeeded on device '{device_str}' with backend '{backend}'"


def detect_jit_capability(device=None, backend: str = "inductor", fullgraph: bool = False,
                          dynamic: Optional[bool] = None, run_probe: bool = True) -> Tuple[bool, str]:
    """Detect whether JIT compilation is achievable on this machine.

    Runs the static environment check first and, if it passes, the functional
    ``torch.compile`` probe.

    Args:
        device (torch.device or str or None, optional): Target device. None infers it
            from the available accelerators.
        backend (str, optional): The ``torch.compile`` backend. Defaults to 'inductor'.
        fullgraph (bool, optional): Whether to require a graph break-free trace. Defaults to False.
        dynamic (bool or None, optional): Dynamic shape handling. Defaults to None.
        run_probe (bool, optional): Set to False to only run the cheap static check.
            Defaults to True.

    Returns:
        tuple[bool, str]: (achievable, reason).
    """
    device_str = resolve_device(device)

    supported, reason = check_jit_support(device_str, backend)
    if not supported or not run_probe:
        return supported, reason

    return probe_jit_compile(device_str, backend, fullgraph, dynamic)


def resolve_jit_enable(configs: Optional[dict], device=None, run_probe: Optional[bool] = None) -> bool:
    """Resolve the user-facing ``compiler_configs['enable']`` flag into a concrete bool.

    The flag accepts:

    * ``'auto'`` (default): detect JIT capability on this machine and gracefully
      fall back to eager mode when it is not achievable.
    * ``True``: force JIT on. The capability check still runs so the user gets a
      warning that explains the upcoming failure, but the request is respected.
    * ``False``: never compile, no detection is performed.

    Args:
        configs (dict or None): The ``compiler_configs`` dict.
        device (torch.device or str or None, optional): Target device.
        run_probe (bool or None, optional): Whether the 'auto' detection runs the functional
            probe on top of the static check. Defaults to None, which follows
            ``configs['auto_probe']`` (itself defaulting to True).

    Returns:
        bool: Whether ``torch.compile`` should be applied.
    """
    configs = configs or {}
    enable = configs.get("enable", "auto")
    if run_probe is None:
        run_probe = bool(configs.get("auto_probe", True))

    if isinstance(enable, str):
        enable = enable.strip().lower()

    if enable is False:
        logger.info("JIT compilation is disabled ('enable': false), running in eager mode")
        return False

    backend = configs.get("backend") or "inductor"
    fullgraph = bool(configs.get("fullgraph", False))
    dynamic = configs.get("dynamic")
    device_str = resolve_device(device)

    if enable is True:
        supported, reason = check_jit_support(device_str, backend)
        if not supported:
            logger.warning(
                f"WARNING: JIT compilation is explicitly enabled ('enable': true) but this machine may not support it: {reason}"
            )
            logger.warning("         Set 'enable': 'auto' to let PtyRAD fall back to eager mode automatically.")
        else:
            logger.info(f"JIT compilation is explicitly enabled ('enable': true) on device '{device_str}'")
        return True

    if enable != "auto":
        logger.warning(f"WARNING: Unrecognized compiler_configs['enable'] = {enable!r}, treating it as 'auto'")

    logger.info(f"### Auto-detecting JIT (torch.compile) capability on device '{device_str}' ###")
    achievable, reason = detect_jit_capability(
        device=device, backend=backend, fullgraph=fullgraph, dynamic=dynamic, run_probe=run_probe
    )

    if achievable:
        logger.info(f"JIT auto-detection passed: {reason}")
        logger.info("-> Enabling JIT compilation for a 1.3-1.9x speedup, the first iteration includes compilation overhead")
    else:
        logger.info(f"JIT auto-detection failed: {reason}")
        logger.info("-> Falling back to eager mode. The reconstruction runs normally, just without the JIT speedup.")
    logger.info(" ")

    return achievable
