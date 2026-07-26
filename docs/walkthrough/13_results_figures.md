# 13. Results and Figures

Demonstrates fine-grained control over which outputs are saved (`save_result`), which figures are generated (`selected_figs`), and output format options (`result_modes` — bit depth, cropped vs. full FOV, 2D/3D/4D).

**When to use:** When disk space is a concern (save only what you need), when preparing publication-ready figures (precise format control), or when automating batch reconstructions that require consistent, specific outputs.

**Tradeoffs & limitations:** Saving too few outputs makes debugging harder and prevents resuming (walkthrough 04 requires `'model'` in `save_result`). The `'raw'` bit option preserves full floating-point precision for downstream analysis; `'8'` produces display-only TIFFs. Choose based on your downstream use. Note that all information is stored in the output model.hdf5.

```{literalinclude} 13_results_figures.yaml
:language: yaml
:linenos:
```
