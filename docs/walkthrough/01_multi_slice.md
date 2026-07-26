# 01. Multi Slice

Splits the object into multiple slices (`obj_Nlayer = 6`, `obj_slice_thickness = 2 Å`) and propagates the probe through them, enabling 3D depth resolution. Depth regularization (`obj_zblur`) is turned on.

**When to use:** Samples thick enough that a single-slice model cannot fit the diffraction data, or when you need depth-resolved (3D) information. Practically all non-2D samples could benefit from using a multislice object to properly model multiple scattering and beam propagation.

**Tradeoffs & limitations:** More free parameters mean slower convergence and higher risk of divergence. The `obj_zblur` depth regularization is practically required — without it, slices tend to collapse to random noise. Total physical thickness must encompass the sample: `obj_Nlayer × obj_slice_thickness`. Note that higher `obj_zblur` regularization provides more stability while reducing depth resolution.

```{literalinclude} 01_multi_slice.yaml
:language: yaml
:linenos:
```
