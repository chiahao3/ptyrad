# 11. Mirrored Amp

Ties object amplitude directly to its phase via `amp = 1 − scale × phase^power`, eliminating amplitude as an independent free parameter and enforcing physical consistency between the two channels.

**When to use:** Electron ptychography in the large-angle ADF regime, where the amplitude–phase relationship is well established. Combining this with positivity (`objp_postiv`) and sparsity (`loss_sparse`) gives the maximum physically constrained reconstruction.

**Tradeoffs & limitations:** Encodes a strong assumption — the relationship must hold across the entire object. Requires `objp_postiv` (phase must be non-negative). If the assumption breaks down (light elements, extreme thicknesses, or large backscatter fractions), the constraint actively degrades quality and obscures quantitative accuracy. The hyperparameters `scale` and `power` are **dataset-dependent** and require tuning.

```{literalinclude} 11_mirrored_amp.yaml
:language: yaml
:linenos:
```
