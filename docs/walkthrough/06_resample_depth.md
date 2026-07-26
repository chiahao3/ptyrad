# 06. Resample Depth

Resamples object slices along z (`obj_z_resample`) to change depth resolution without restarting. Can be chained with `obj_z_pad`: padding is always applied first, then resampling.

**When to use:** After a coarse multislice run, to refine depth resolution by increasing the number of slices while keeping the same total physical thickness. Useful to restart a finer reconstruction from a converged coarser one.

**Tradeoffs & limitations:** Upsampling interpolates existing data — it does not create independent new information. The actual resolvable depth is still limited by the dataset, not the slice count. Finer slices still require `obj_zblur` depth regularization, now at a proportionally smaller `std`, as the blur kernel is specified in unit of number of slices.

```{literalinclude} 06_resample_depth.yaml
:language: yaml
:linenos:
```
