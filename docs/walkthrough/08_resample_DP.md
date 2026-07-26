# 08. Resample DP

Upsamples diffraction patterns on the fly (or precomputed) via `meas_resample`, increasing the detector pixel count (`Npix`) without changing `kMax`, which adds real-space padding around the probe in the reconstruction.

**When to use:** Required when thick samples, large defocus, or high convergence angles cause probe intensity to spread near the edge of the diffraction pattern — the FFT's periodic boundary then wraps intensity around and corrupts the reconstruction. Upsampling is the standard approach; `simu_Npix` is a theoretically cleaner alternative but can suffer from optimization instability.

**Tradeoffs & limitations:** A 2× scale factor quadruples memory and compute per pattern. Unlike `meas_pad`, resampling does not change the physical scattering angle range. For thin samples with well-contained probes, it adds overhead without benefit.

```{literalinclude} 08_resample_DP.yaml
:language: yaml
:linenos:
```
