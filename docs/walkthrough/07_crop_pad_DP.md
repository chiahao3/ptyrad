# 07. Crop and Pad DP

Preprocesses diffraction patterns at initialization: `meas_crop` removes zero-padded borders or selects a sub-region in k-space (or real space), and `meas_pad` restores the target pixel count using a power-law padding scheme.

**When to use:** When the raw 4D-STEM dataset contains unnecessary zero-padded borders that waste memory and compute, or when a specific real-space ROI is desired. PtyRAD automatically recalculates geometry after cropping.

**Tradeoffs & limitations:** Cropping and padding the diffraction pattern changes `kMax` (the maximum collected scattering angle) and therefore the reconstruction resolution ceiling. Power-law padding is an approximation; for clean bandwidth-limited interpretation, prefer zero padding. Aggressive padding can introduce artifacts.

## Optional scattering prior for a simulated exterior

For a 128-pixel measured diffraction pattern and a 256-pixel simulation, set `simu_Npix: 256` and `simu_match_mode: crop`. Leave `meas_pad` disabled. The default data losses still use the measured center. Add the following to the existing `loss_params` block to enable the exterior prior.

```yaml
loss_outside:
  state: true
  weight: 0.05
  transition_px: 4.0
  margin: 1.25
```

The prior fits a radial intensity curve `B + A * ((1 + s²) / (r² + s²))²` to the mean diffraction intensity in measured diagonal sectors. Here `r` is the radius in units of the distance to the detector edge and `s` is the fitted screening scale. The elastic term is inspired by the first Born approximation for a screened Coulomb potential. The measured detector corners beyond the inscribed circle calibrate the allowance, which is further increased by `margin`.

Only simulated pixels outside the crop enter this term. Their weights rise smoothly with distance from the edge over `transition_px` detector pixels. Within each minibatch, the term pools intensity from the exterior into annuli and penalizes its squared amplitude excess above the bound. The calibration and weighted term are logged as `loss_outside`. The bound is an empirical prior on radial averages. It supplies no measurements beyond the detector and does not set a target for individual Bragg spots. Use centered diffraction data on a square detector and compare with the same crop run with the prior disabled.

```{literalinclude} 07_crop_pad_DP.yaml
:language: yaml
:linenos:
```
