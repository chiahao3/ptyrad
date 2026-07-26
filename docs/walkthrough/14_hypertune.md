# 14. Hypertune

Activates PtyRAD's integrated Bayesian hyperparameter search (via Optuna) through `hypertune_params`, automatically exploring a defined parameter space across multiple short reconstruction trials.

**When to use:** When key parameters such as overfocus (`C10`) or number of slices (`Nlayer`) are uncertain and manual grid search would be expensive. Limit to 2–4 parameters at a time; results are stored in an SQLite database compatible with Optuna Dashboard for interactive analysis.

**Tradeoffs & limitations:** Total cost is `n_trials × NITER` reconstructions. Setting `NITER` too low gives noisy trial metrics; too high wastes compute. The search is probabilistic and 50 trials may not find the global optimum in a large space. Intermediate checkpoints are disabled (`SAVE_ITERS: null`) to reduce I/O overhead per trial.

```{literalinclude} 14_hypertune.yaml
:language: yaml
:linenos:
```
