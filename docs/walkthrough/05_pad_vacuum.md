# 05. Pad Vacuum

Inserts vacuum (empty) slices at the top and/or bottom of the multislice object via `obj_z_pad`, extending the depth window and shifting the probe entrance plane. `probe_z_shift` and `obj_z_recenter` are used in tandem.

**When to use:** When the sample is not centered in the initial depth window, or when you want to extend the reconstruction volume after an initial run without restarting. Always pair with `probe_z_shift` to preserve the sample–probe relative alignment.

**Tradeoffs & limitations:** Each vacuum layer increases total slice count and therefore compute cost. The `probe_z_shift` value must be consistent with the added padding (`n_pad_layers × obj_slice_thickness`). Getting this wrong shifts the probe relative to the sample.

```{literalinclude} 05_pad_vacuum.yaml
:language: yaml
:linenos:
```
