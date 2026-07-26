# 00. Single Slice

The simplest configuration: a single-layer (2D projected potential) object with mixed-state probe modes — the recommended starting point for any new dataset.

**When to use:** Thin samples where depth resolution is not needed, or as a first run to verify data quality and calibration before enabling multi-slice. For thicker samples (~20 nm), this can still provide quick qualitative feedback.

**Tradeoffs & limitations:** All depth structure is collapsed into one 2D projection, so axial information is lost. This is not a limitation for truly thin specimens, but will produce inaccurate reconstructions for samples thick enough that dynamical channeling across slices matters.

```{literalinclude} 00_single_slice.yaml
:language: yaml
:linenos:
```
