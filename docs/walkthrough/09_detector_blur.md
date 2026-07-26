# 09. Detector Blur

Models the detector's point spread function (PSF) by applying a Gaussian blur (in k-space pixels) to the simulated forward diffraction patterns via `detector_blur_std`.

**When to use:** When the required number of mixed-state probe modes to fit the data is unreasonably large, or when using a thick scintillator detector with a known broad PSF. A well-matched detector blur can reduce the needed probe modes and speed up reconstruction.

**Tradeoffs & limitations:** The PSF is assumed to be Gaussian, which is an approximation. Too large a `detector_blur_std` smooths out fine diffuse scattering and reduces sensitivity to high-frequency features. Typical values are 0–1 px; validate by visually comparing forward-modeled patterns against experimental data.

```{literalinclude} 09_detector_blur.yaml
:language: yaml
:linenos:
```
