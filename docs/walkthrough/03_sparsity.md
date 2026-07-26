# 03. Sparsity

Adds L1-norm sparsity regularization (`loss_sparse`) on top of positivity, promoting discrete atomic-column-like reconstructions.

**When to use:** Crystalline specimens with well-separated atomic columns where sharper, higher-contrast results are desired. Introduce after a stable positivity run; start with small weights (0.01–0.1) and increase gradually.

**Tradeoffs & limitations:** Must be combined with the positivity constraint. Over-weighting can create artifacts and suppress real signal. Note that sparsity + positivity is designed to clean up low phase values, so for quantitative analysis start from unconstrained results.

```{literalinclude} 03_sparsity.yaml
:language: yaml
:linenos:
```
