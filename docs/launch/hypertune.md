# Hypertune mode

Hyperparameter tuning is essentially running many individual reconstuction jobs with different parameters, and form a statistical model given the outputs to predict what combination of parameters would give the best reconstruction. In *PtyRAD*, the hyperparameter tuning is called **"hypertune mode"** and is powered by [Optuna](https://optuna.readthedocs.io/en/stable/), which is one of the most popular hyperparameter tuning framework in machine learning community.

Running hypertune with a single GPU is not ideal because each reconstruction task will be running sequentially. *PtyRAD* provides a frictionless setup to distribute the hypertune task over multiple GPUs as shown below.

## Hypertune mode on multiple GPUs

Assuming you're using Slurm on a HPC, the setup is nearly identical with the previous example, except that you provide a params file with `hypertune_params` configured and `if_hypertune: true`.

```bash
# This is your `slurm_run_ptyrad_hypertune.sub` job script

## Set the params_path variable
PARAMS_PATH="params/walkthrough/14_hypertune.yaml"
echo params_path = ${PARAMS_PATH}
```

Go to `ptyrad/` root directory, and execute the following command:

```bash
# This will submit the same Slurm job (`slurm_run_ptyrad_hypertune.sub`) 5 times, 
# each job will be running on 1 GPU
bash ./scripts/LoopSubmit.sh -n 5 
```

> 💡 Note: Params file must be able to locate target files (e.g. measurements) from your directory during submission. For robustness, use absolute path.