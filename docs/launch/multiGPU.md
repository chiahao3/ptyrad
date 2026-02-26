# Multiple GPUs

Hypertune mode distributes many individual reconstruction tasks on multiple GPUs, where each task utilizes exactly 1 GPU. *PtyRAD* also supports running the **reconstuction mode with multiple GPUs**, which split the workload of a single large reconstuction task on multiple GPUs. *PtyRAD* uses the convenient API wrapper from *HuggingFace Accelerate* that wraps around PyTorch's Distributed Data Parallel (DDP) framework.

## Reconstruciton mode on multiple GPUs

To reconstruct with multiple GPUs, you'll need the following:

- Linux system
- more than 1 NVIDIA GPU on the same machine (multi-instance GPU, or MIGs are NOT considered as multi GPU and would not work, see [here](https://discuss.pytorch.org/t/parallel-training-with-invidia-migs/159445).)
- `accelerate` package from HuggingFace (it's currently listed as a core dependency of PtyRAD so if you follow the README instruction while creating environment, it should already be included)

> 💡 This is the same example as `ptyrad/scripts/slurm_run_ptyrad_multiGPU.sub`.

```bash
#!/bin/bash
#SBATCH --job-name=multiGPU
#SBATCH --mail-user=cl2696@cornell.edu       # Where to send mail
#SBATCH --nodes=1                            # number of nodes requested
#SBATCH --ntasks=1                           # number of tasks to run in parallel
#SBATCH --cpus-per-task=64                    # number of CPUs required for each task. 4 for 10GB, 8 for 20GB, 32 for 80GB of A100.
#SBATCH --gres=gpu:a100:2               # request a GPU #gpu:a100:1, gpu:2g.20gb:1
#SBATCH --time=168:00:00                      # Time limit hrs:min:sec
#SBATCH --output=log_job_%j_ptyrad_PSO_multiGPU.txt  # Standard output and error log

## Assuming you are under root `ptyrad/` working directory and calling `sbatch scripts/slurm_run_ptyrad_multiGPU.sub`
pwd; hostname; date

module load cuda/11.8

source activate ptyrad

## Set the params_path variable
## Make sure the path specified inside params.yml is reachable from your root (i.e., `ptyrad/`)
PARAMS_PATH="params/examples/PSO.yml"
echo params_path = ${PARAMS_PATH}

## The gpuid is used to assign the device for PtyRAD, it can be either 'acc', 'cpu', or an integer
## The jobid is used as a unique identifier for hypertune mode with multiple GPU workers on different nodes. 
## The JOBID is an environment variable that'll be automatically set to 1-N via LoopSubmit.sh. If not set, default to 0.

## Execute the ptyrad CLI command `ptyrad run`
# ptyrad run "${PARAMS_PATH}" --gpuid 0 --jobid "${JOBID:-0}" 2>&1 # This runs via ptyrad CLI command on 1 GPU. 

## For multi-GPU and mixed-precision, explicitly pass `--gpuid acc` so that we can defer the device assignment to the "HuggingFace accelerate" package
## These capabilities are only available while launched via `accelerate``, and are only supported on non-MIG nodes so we can only do c0001 (full A100s)
## On A100, mixed-precision does not give significant speedup because A100 uses mostly tensor cores for TF32 dtype
## multi-GPU can also be much slower if the batch size is too small. Typically smaller batch sizes converge faster unless we scale the learning rate with batch size.
accelerate launch --multi_gpu --num_processes=2 -m ptyrad run "${PARAMS_PATH}" --gpuid acc 2>&1 # This runs DDP on 2 GPUs

date
```

> 💡 Note that 2 GPUs are requested `--gres=gpu:a100:2`, and the `ptyrad run` command is launched via `accelerate launch --multi_gpu`.