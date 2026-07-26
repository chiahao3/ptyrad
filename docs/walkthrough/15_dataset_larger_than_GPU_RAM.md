# 15. Dataset Larger Than GPU RAM

Shows how to enable asynchronous streaming (`preload_data: false`) so that the full diffraction dataset stays in CPU RAM and only the current mini-batch is transferred to GPU per iteration.

**When to use:** When the 4D-STEM dataset is too large to fit in GPU VRAM. Requires only a single flag change — no other configuration modification is needed.

**Tradeoffs & limitations:** CPU→GPU transfer adds latency each iteration. With asynchronous loading the overhead is small but non-zero; for datasets that fit in GPU memory, `preload_data: true` is often faster.

```{literalinclude} 15_dataset_larger_than_GPU_RAM.yaml
:language: yaml
:linenos:
```
