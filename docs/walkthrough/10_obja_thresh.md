# 10. Obja Thresh

Constrains object amplitude to a narrow range around 1 via `obja_thresh`, reflecting the physical expectation that most forward-scattered electrons are transmitted and collected by the detector.

**When to use:** For datasets where the object amplitude is expected to be close to 1 (large collection angle, ADF-like regime). Especially important for multislice reconstructions: amplitude deviations are multiplicative, so a per-slice amplitude of 0.95 becomes ~0.60 after 10 layers.

**Tradeoffs & limitations:** Too tight a threshold suppresses real scattering contrast at heavy-atom sites. The `relax` parameter softens the boundary. Specimens with heavy atoms or thick unit cells need looser bounds; use the threshold range as a physically motivated guard rather than a hard constraint.

```{literalinclude} 10_obja_thresh.yaml
:language: yaml
:linenos:
```
