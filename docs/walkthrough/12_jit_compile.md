# 12. JIT Compile

Controls PyTorch's JIT (just-in-time) compiler via `compiler_configs: {enable: ...}`, which fuses and optimizes GPU kernels at runtime for a measured 1.3–1.9× speedup.

`enable` accepts three values:

- `'auto'` (default) — right before the reconstruction loop starts, PtyRAD checks whether JIT compilation is achievable on the current machine (PyTorch version, TorchDynamo/TorchInductor support, compute device, CUDA compute capability, Triton or C++ compiler availability) and then compiles and runs a tiny function as a smoke test to confirm the toolchain actually works. JIT is used when the check passes, otherwise PtyRAD logs the reason and falls back to eager mode. **macOS is deliberately excluded from `'auto'`** — see below.
- `true` — always compile, on any platform. The capability check still runs and warns when the machine looks unsupported, so failures surface instead of being silently skipped.
- `false` — never compile.

**macOS is opt-in:** Triton / TorchInductor has been noticeably less stable on macOS than on Linux and Windows, including cases where a compiled kernel returns incorrect numbers rather than failing outright. A silently wrong reconstruction is much worse than a slower one, so `'auto'` keeps macOS in eager mode and Mac users turn JIT on deliberately with `enable: true`. If you do, compare a short run against eager before trusting the results.

**When to use:** `'auto'` is the recommended setting for essentially all runs — it gives the speedup wherever the hardware permits and stays out of the way where it does not. Use `true` when benchmarking or debugging the compiled path and you want a hard failure rather than a fallback; use `false` to force eager mode, e.g. when comparing against a compiled run.

**Tradeoffs & limitations:** The first epoch incurs a compilation overhead before the speedup takes effect, and the `'auto'` smoke test adds a one-time compile of a tiny function (cached per process, and reused across Optuna trials in hypertune mode). JIT on CUDA GPUs requires Triton and compute capability ≥ 7.0; on Windows that means the `triton-windows` package. JIT on Apple Silicon (MPS) requires PyTorch ≥ 2.7 for the TorchInductor Metal backend, JIT on Windows requires PyTorch ≥ 2.7 (earlier versions report TorchInductor as unsupported there), and JIT on CPU requires a C++ compiler on `PATH`. Speedup follows a complicated scaling law with problem size (`Npix`, probe modes, slice count, batch sizes) — small problems see less benefit.

```{literalinclude} 12_jit_compile.yaml
:language: yaml
:linenos:
```
