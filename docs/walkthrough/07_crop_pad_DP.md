# 07. Crop and Pad DP

Preprocesses diffraction patterns at initialization: `meas_crop` removes zero-padded borders or selects a sub-region in k-space (or real space), and `meas_pad` restores the target pixel count using a power-law padding scheme.

**When to use:** When the raw 4D-STEM dataset contains unnecessary zero-padded borders that waste memory and compute, or when a specific real-space ROI is desired. PtyRAD automatically recalculates geometry after cropping.

**Tradeoffs & limitations:** Cropping and padding the diffraction pattern changes `kMax` (the maximum collected scattering angle) and therefore the reconstruction resolution ceiling. Power-law padding is an approximation; for clean bandwidth-limited interpretation, prefer zero padding. Aggressive padding can introduce artifacts.

```{literalinclude} 07_crop_pad_DP.yaml
:language: yaml
:linenos:
```
