# 04. Resume Reconstruction

Demonstrates loading a previous reconstruction (object, probe, scan positions) from an HDF5 output file to continue refining it, rather than restarting from scratch.

**When to use:** To incrementally add constraints, increase complexity (e.g. add more slices), or recover from an interrupted run. Also useful for mixing results across software (e.g. an object from PtychoShelves with a probe from PtyRAD).

**Tradeoffs & limitations:** Output folder names must be managed carefully via `prefix`/`postfix` to avoid overwriting prior results. Source file paths and array shapes must match; mismatched scan geometries or slice counts will raise errors at load time.

```{literalinclude} 04_resume_reconstruction.yaml
:language: yaml
:linenos:
```
