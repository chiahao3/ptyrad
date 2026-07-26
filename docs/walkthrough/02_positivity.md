# 02. Positivity

Adds a positivity constraint (`objp_postiv`) on top of the multi-slice setup, iteratively clipping negative phase values to zero.

**When to use:** The projected potential is expected to be positive, so a positivity constraint creates a natural baseline for the reconstructed phase. Use this when you want higher visual contrast and easier comparison to calculated atomic potentials. Apply after a stable unconstrained run; you can also disable it on a subsequent run to recover a quantitative phase baseline.

**Tradeoffs & limitations:** Enhances contrast but could reduce phase quantitativeness — the clipping imposes a physical assumption (non-negative potential) that could remove the weak tails of the reconstructed phase from light elements, especially when used simultaneously with `loss_sparse`. Use the `relax` parameter for a softer boundary.

```{literalinclude} 02_positivity.yaml
:language: yaml
:linenos:
```
