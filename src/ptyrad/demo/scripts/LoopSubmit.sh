#!/bin/bash

# This shell script is designed to submit multiple jobs for PtyRAD in Hypertune modes
# So we can utilize multiple GPU workers
# Go to the `demo/` root directory and call this script with `bash ./scripts/LoopSubmit.sh`

# Define parameters
# No spaces around '=' in variable assignments
# Default N = 5, but can be updated by passing "-n $N" in the command-line argument
SUBDIR="./scripts"
N=5
script="slurm_run_ptyrad.sub"

# Parse the command-line argument for -n
while getopts n: flag
do
    case "${flag}" in
        n) N=${OPTARG};;
    esac
done

# Loop to submit the sbatch script N times, passing the job number (i) as an environment variable
for i in $(seq 1 $N); do
    echo "Submitting $SUBDIR/$script with jobid $i"
    export JOBID=$i  # Export jobid as an environment variable
    sbatch "$SUBDIR/$script"
    if [ "$i" -eq 1 ]; then
        sleep 60
    else
        sleep 10
    fi
done

echo "Jobs are all submitted!"
