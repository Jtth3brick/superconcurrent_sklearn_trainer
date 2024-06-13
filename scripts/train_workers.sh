#!/bin/bash
#SBATCH --job-name=train_workers
#SBATCH --account=ac_wolflab
#SBATCH --partition=savio4_htc
#SBATCH --array=0-499
#SBATCH --time=72:00:00

PYTHON=$(which python)

${PYTHON} train.py "work" --worker_id $SLURM_ARRAY_TASK_ID
