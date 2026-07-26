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
2. A *functional* smoke test that actually compiles and runs a tiny function
   (forward + backward) on the target device, which is the only reliable way
   to catch broken toolchains that look fine on paper.

Both stages are cached per-process, so repeated calls (e.g. one per Optuna
trial during hypertune) pay the cost only once.

On top of raw capability, 'auto' also applies an opt-in policy: macOS keeps
JIT off by default because Triton / TorchInductor has been less stable there
(see `AUTO_OPT_IN_SYSTEMS`), so Mac users enable it deliberately with
``{'enable': true}``.

Every PyTorch API used here exists in PyTorch 2.4, PtyRAD's declared minimum,
so this module needs no version bump. The private ones
(`torch._dynamo.is_dynamo_supported`, `torch._dynamo.is_inductor_supported`,
`torch.utils._triton.has_triton_package`) are additionally wrapped so that a
build without them degrades to the smoke test instead of raising. Note that
PyTorch < 2.7 reports TorchInductor as unsupported on Windows, in which case
'auto' correctly stays in eager mode there.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import platform
import shutil
import warnings
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# torch.compile was introduced in PyTorch 2.0. This is only the floor for a usable
# torch.compile, PtyRAD's install requirement (torch>=2.4) is stricter anyway.
MIN_TORCH_VERSION = (2, 0)

# Triton (the GPU backend used by TorchInductor) requires CUDA compute capability >= 7.0 (Volta)
MIN_CUDA_CAPABILITY = (7, 0)

# TorchInductor gained a Metal (MPS) codegen backend in PyTorch 2.7,
# earlier versions raise or silently degrade when compiling on Apple Silicon GPUs
MIN_TORCH_VERSION_MPS = (2, 7)

# Keys of `compiler_configs` that belong to PtyRAD and must not reach torch.compile
PTYRAD_ONLY_CONFIG_KEYS = ("enable", "auto_smoke_test")

# Device types whose TorchInductor codegen path goes through Triton
TRITON_DEVICE_TYPES = ("cuda", "xpu", "hpu")

# Platforms where JIT is capable but not trusted enough to switch on by itself.
# Triton / TorchInductor on macOS has been less stable than on Linux and Windows, including
# reports of silently incorrect numerics, and a wrong reconstruction is far worse than a slow
# one. So 'auto' stays in eager mode on macOS and users opt in with an explicit 'enable': true.
AUTO_OPT_IN_SYSTEMS = ("Darwin",)

MACOS_JIT_CAVEAT = (
    "Triton / TorchInductor has been less stable on macOS than on Linux and Windows, "
    "occasionally producing incorrect numerical results rather than failing outright."
)

AUTO_OPT_IN_REASON = (
    f"JIT is not enabled automatically on macOS. {MACOS_JIT_CAVEAT} "
    "Set 'enable': true to opt in explicitly."
)

OPT_IN_ACCEPTED_WARNING = (
    f"JIT is explicitly enabled on macOS. {MACOS_JIT_CAVEAT} "
    "Please sanity-check this reconstruction against an eager run ('enable': false)."
)

TRITON_WINDOWS_HINT = (
    "Triton does not officially support Windows. "
    "Install `triton-windows` (https://github.com/woct0rdho/triton-windows) to enable JIT on Windows."
)


def _system() -> str:
    """Return the OS name ('Linux', 'Darwin', 'Windows').

    Wrapped in a helper so tests can simulate another OS without patching `platform.system`
    itself, which PyTorch also calls while importing (patching it globally breaks that import).
    """
    return platform.system()


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
    only gave a bare 'cuda') so that the capability check and the functional smoke test both
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


def _has_triton_package() -> bool:
    """Check whether the Triton package is installed.

    Note:
        This deliberately checks package availability only, not device compatibility.
        `torch.utils._triton.has_triton()` would also validate the device, but it inspects
        the *current* CUDA device rather than a given one (and caches its verdict), so on a
        heterogeneous host it answers for the wrong GPU. The per-device part is covered by
        `_cuda_capability()` and, ultimately, by the smoke test on the selected device.
    """
    try:
        from torch.utils._triton import has_triton_package

        return bool(has_triton_package())
    except Exception:
        # Fall back to a plain import check on older PyTorch
        try:
            return importlib.util.find_spec("triton") is not None
        except Exception:
            return False


def _has_cxx_compiler() -> bool:
    """Check whether a C++ compiler is on PATH (needed by TorchInductor's CPU backend)."""
    if _system() == "Windows":
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

    system = _system()
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
        pass  # Too old or trimmed-down build, let the functional smoke test be the judge

    # (3) Backend-specific requirements
    if backend == "inductor":
        try:
            from torch._dynamo import is_inductor_supported

            if not is_inductor_supported():
                return False, f"TorchInductor is unsupported by PyTorch {torch.__version__} on this platform"
        except ImportError:
            pass

        if device_type in TRITON_DEVICE_TYPES:
            if not _has_triton_package():
                hint = f" {TRITON_WINDOWS_HINT}" if system == "Windows" else ""
                return False, (
                    f"The `triton` package needed by the '{device_type}' TorchInductor backend is not installed.{hint}"
                )

            if device_type == "cuda":
                if not torch.cuda.is_available():
                    return False, f"PyTorch reports no available CUDA device for '{device_str}'"

                # Read the capability of the selected GPU only, so an older GPU elsewhere
                # on a heterogeneous host doesn't veto JIT on the one actually in use
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


def _jit_smoke_test_fn(x):
    """Tiny function compiled by the JIT smoke test (kept at module level so Dynamo can trace it)."""
    return (x * x + 1.0).sum()


def _sync_device(device) -> None:
    """Synchronize the device so asynchronous kernel failures surface inside the smoke test."""
    import torch

    try:
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        elif device.type == "mps" and torch.backends.mps.is_available():
            torch.mps.synchronize() # As of pytorch 2.10, torch.mps.synchronize doesn't take any arg
    except Exception:
        pass


@contextmanager
def _preserved_rng_state(device):
    """Restore every RNG the smoke test could touch, so detection never shifts the user's stream.

    The smoke test itself draws no random numbers, but TorchInductor does (`mode='max-autotune'`
    benchmarks candidate kernels on random inputs). Since the result is cached, one run would
    otherwise consume RNG that a second, identically seeded reconstruction in the same process
    would not, silently desynchronizing anything downstream that samples (e.g. the random
    subsampling inside `approx_torch_quantile`).
    """
    import torch

    states = {}
    try:
        states["cpu"] = torch.get_rng_state()
        if device.type == "cuda" and torch.cuda.is_available():
            states["cuda"] = torch.cuda.get_rng_state(device)
        elif device.type == "mps" and torch.backends.mps.is_available() and hasattr(torch.mps, "get_rng_state"):
            states["mps"] = torch.mps.get_rng_state()
    except Exception:
        pass # Best effort, a missing snapshot must never block the smoke test

    try:
        yield
    finally:
        try:
            if "cpu" in states:
                torch.set_rng_state(states["cpu"])
            if "cuda" in states:
                torch.cuda.set_rng_state(states["cuda"], device)
            if "mps" in states:
                torch.mps.set_rng_state(states["mps"])
        except Exception:
            pass


def _short(err, max_chars: int = 300) -> str:
    """Trim an exception message for logging, since some backend errors dump every known option."""
    message = " ".join(str(err).split())
    if len(message) > max_chars:
        message = f"{message[:max_chars]}... (truncated, re-run with --verbosity DEBUG for the full error)"
    return message


def _freeze(value):
    """Make a compile-kwargs value hashable so it can key the smoke-test cache."""
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, set, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


# Cache keyed by (device, effective compile kwargs). A plain dict rather than lru_cache because
# the kwargs carry an unhashable 'options' dict that torch.compile needs back as a real dict.
_SMOKE_TEST_CACHE: dict = {}


def smoke_test_jit_compile(device_str: str = "cpu", **compile_kwargs) -> Tuple[bool, str]:
    """Functionally verify ``torch.compile`` by compiling and running a tiny function.

    This actually exercises the whole toolchain (Dynamo trace, backend codegen,
    kernel build, forward, and backward) on the target device with the *same* kwargs
    the reconstruction will use, which is the only reliable way to catch environments
    that pass the static check but have a broken compiler, missing CUDA headers, an
    incompatible Triton build, or an unusable backend option.

    The smoke test compiles a 256 element function, so the cost is dominated by the
    one-time backend warmup (a few seconds at most) and is cached per-process. It draws
    no random numbers and restores any RNG state the backend consumes.

    Note:
        A hard crash (segfault) inside a broken backend cannot be caught here,
        only Python-level exceptions are handled.

    Args:
        device_str (str, optional): Target device, e.g. 'cpu', 'mps', or 'cuda:1'. The smoke test runs
            on the exact device given so it reflects the GPU used by the reconstruction.
            Defaults to 'cpu'.
        **compile_kwargs: The kwargs forwarded to ``torch.compile`` (backend, fullgraph, dynamic,
            mode, options, ...). Any 'disable' entry is ignored, since a disabled compile would
            make the smoke test vacuous.

    Returns:
        tuple[bool, str]: (works, reason) describing whether the smoke test compiled and ran cleanly.
    """
    import torch

    compile_kwargs.pop("disable", None)
    cache_key = (device_str, _freeze(compile_kwargs))
    if cache_key in _SMOKE_TEST_CACHE:
        return _SMOKE_TEST_CACHE[cache_key]

    result = _run_jit_smoke_test(device_str, compile_kwargs)
    _SMOKE_TEST_CACHE[cache_key] = result
    return result


smoke_test_jit_compile.cache_clear = _SMOKE_TEST_CACHE.clear


def _run_jit_smoke_test(device_str: str, compile_kwargs: dict) -> Tuple[bool, str]:
    """Uncached body of `smoke_test_jit_compile`."""
    import torch

    try:
        device = torch.device(device_str)
    except Exception as err:
        return False, f"invalid device type '{device_str}' ({err})"

    try:
        with warnings.catch_warnings(), _preserved_rng_state(device):
            warnings.simplefilter("ignore")

            # A deterministic spread of values, so the smoke test consumes no RNG of its own
            x = torch.linspace(-1.0, 1.0, 256, device=device).reshape(16, 16).requires_grad_(True)
            compiled_fn = torch.compile(_jit_smoke_test_fn, **compile_kwargs)
            out = compiled_fn(x)
            out.backward()
            _sync_device(device)

            # Guard against a backend that "succeeds" but returns garbage
            if x.grad is None or not bool(torch.isfinite(out).all()) or not bool(torch.isfinite(x.grad).all()):
                return False, "the compiled smoke-test function returned non-finite values"

    except Exception as err:
        # torch.compile failures surface as BackendCompilerFailed, Unsupported, ImportError, OSError, RuntimeError...
        logger.debug(f"Full torch.compile smoke test error: {err}", exc_info=True)
        return False, f"torch.compile smoke test failed with {type(err).__name__}: {_short(err)}"
    finally:
        # Leave no compiled state behind, the reconstruction loop resets and compiles its own graphs
        try:
            torch._dynamo.reset()
        except Exception:
            pass

    return True, (
        f"torch.compile smoke test succeeded on device '{device_str}' with {compile_kwargs or 'default configs'}"
    )


def detect_jit_capability(device=None, compile_kwargs: Optional[dict] = None,
                          run_smoke_test: bool = True) -> Tuple[bool, str]:
    """Detect whether JIT compilation is achievable on this machine.

    Runs the static environment check first and, if it passes, the functional
    ``torch.compile`` smoke test with the same kwargs the reconstruction will use.

    This answers the question behind ``'enable': 'auto'``, so on top of raw capability it
    also applies PtyRAD's opt-in policy: platforms in `AUTO_OPT_IN_SYSTEMS` are reported as
    not achievable even when they could compile. Use `check_jit_support` instead for the
    pure capability verdict.

    Args:
        device (torch.device or str or None, optional): Target device. None infers it
            from the available accelerators.
        compile_kwargs (dict or None, optional): The effective ``torch.compile`` kwargs
            (backend, fullgraph, dynamic, mode, options, ...). Defaults to None, meaning
            torch.compile defaults.
        run_smoke_test (bool, optional): Set to False to only run the cheap static check.
            Defaults to True.

    Returns:
        tuple[bool, str]: (achievable, reason).
    """
    if _system() in AUTO_OPT_IN_SYSTEMS:
        return False, AUTO_OPT_IN_REASON

    compile_kwargs = dict(compile_kwargs or {})
    device_str = resolve_device(device)

    supported, reason = check_jit_support(device_str, compile_kwargs.get("backend") or "inductor")
    if not supported or not run_smoke_test:
        return supported, reason

    return smoke_test_jit_compile(device_str, **compile_kwargs)


def compile_kwargs_from_configs(configs: Optional[dict]) -> dict:
    """Turn user-facing `compiler_configs` into the kwargs `torch.compile` actually accepts.

    Drops the PtyRAD-only keys and resolves the one combination the params schema allows but
    torch.compile rejects: passing both 'mode' and 'options' raises, so the more specific
    'options' wins. Both the smoke test and the reconstruction compile with these kwargs, so
    a bad option is caught by detection instead of at the first iteration.

    Args:
        configs (dict or None): The user-facing `compiler_configs` dict.

    Returns:
        dict: The effective `torch.compile` kwargs.
    """
    kwargs = {k: v for k, v in (configs or {}).items() if k not in PTYRAD_ONLY_CONFIG_KEYS}

    if kwargs.get("options") is not None and kwargs.get("mode") is not None:
        dropped = kwargs.pop("mode")
        logger.info(
            f"compiler_configs sets both 'mode' ({dropped!r}) and 'options', but torch.compile accepts "
            "only one of them, so 'options' is used and 'mode' is ignored"
        )

    return kwargs


def resolve_jit_enable(configs: Optional[dict], device=None, run_smoke_test: Optional[bool] = None,
                       compile_kwargs: Optional[dict] = None) -> bool:
    """Resolve the user-facing ``compiler_configs['enable']`` flag into a concrete bool.

    The flag accepts:

    * ``'auto'`` (default): detect JIT capability on this machine and gracefully
      fall back to eager mode when it is not achievable, or when the platform is
      opt-in only (macOS, see `AUTO_OPT_IN_SYSTEMS`).
    * ``True``: force JIT on. The capability check still runs so the user gets a
      warning that explains the upcoming failure, but the request is respected.
    * ``False``: never compile, no detection is performed.

    Args:
        configs (dict or None): The ``compiler_configs`` dict.
        device (torch.device or str or None, optional): Target device.
        run_smoke_test (bool or None, optional): Whether the 'auto' detection runs the functional
            smoke test on top of the static check. Defaults to None, which follows
            ``configs['auto_smoke_test']`` (itself defaulting to True).
        compile_kwargs (dict or None, optional): Pre-computed effective ``torch.compile`` kwargs,
            so a caller that already built them doesn't derive (and log about) them twice.
            Defaults to None, which derives them from `configs`.

    Returns:
        bool: Whether ``torch.compile`` should be applied.
    """
    configs = configs or {}
    enable = configs.get("enable", "auto")
    if run_smoke_test is None:
        run_smoke_test = bool(configs.get("auto_smoke_test", True))
    if compile_kwargs is None:
        compile_kwargs = compile_kwargs_from_configs(configs)

    if isinstance(enable, str):
        enable = enable.strip().lower()

    if enable is False:
        logger.info("JIT compilation is disabled ('enable': false), running in eager mode")
        return False

    backend = compile_kwargs.get("backend") or "inductor"
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

        if _system() in AUTO_OPT_IN_SYSTEMS:
            logger.warning(f"WARNING: {OPT_IN_ACCEPTED_WARNING}")
        return True

    if enable != "auto":
        logger.warning(f"WARNING: Unrecognized compiler_configs['enable'] = {enable!r}, treating it as 'auto'")

    logger.info(f"### Auto-detecting JIT (torch.compile) capability on device '{device_str}' ###")
    achievable, reason = detect_jit_capability(
        device=device, compile_kwargs=compile_kwargs, run_smoke_test=run_smoke_test
    )

    if achievable:
        logger.info(f"JIT auto-detection passed: {reason}")
        logger.info("-> Enabling JIT compilation for a 1.3-1.9x speedup, the first iteration includes compilation overhead")
    else:
        logger.info(f"JIT auto-detection failed: {reason}")
        logger.info("-> Falling back to eager mode. The reconstruction runs normally, just without the JIT speedup.")
    logger.info(" ")

    return achievable
