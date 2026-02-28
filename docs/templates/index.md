(reference:templates)=

# Templates

This section contains PtyRAD template params files with different levels of control. 

Choosing the right template depends on your familiarity with the software and the complexity of your dataset. 

We generally recommend starting simple and only exposing more parameters as your reconstruction requires them.

* **{doc}`Minimal <minimal>`:** The absolute bare essentials needed to run a reconstruction. Best for quick tests, tutorials, or if you want to rely entirely on PtyRAD's default behaviors.

* **{doc}`Standard (Recommended) <standard>`:** The sweet spot for most users. It exposes the most commonly tuned parameters (like basic constraints, and learning rates) without overwhelming you with niche settings. Start here for standard datasets.

* **{doc}`Advanced <advanced>`:** Unlocks finer controls for challenging data. Use this if you need to tweak advanced data preprocessing, more advanced constraints, or output preference settings.

* **{doc}`Full <full>`:** An exhaustive list of every single parameter available in PtyRAD. Best used as a reference dictionary rather than a daily driver, or for power users who need absolute granular control over the entire pipeline including hypertune.

```{toctree}
:maxdepth: 1
:hidden:

minimal
standard
advanced
full
```