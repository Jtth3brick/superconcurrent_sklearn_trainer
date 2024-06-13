#!/bin/bash
#SBATCH --job-name=train_manager
#SBATCH --account=ac_wolflab
#SBATCH --partition=savio3_htc
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G

PYTHON=$(which python)

${PYTHON} train.py "manage" --cv=true --validate=true --pipeline_names "lasso" "ridge" "enet" "svc" "xgb" "rf" "nn" --split_names "U1" "U2" "A1" "A2" "B1" "B2" "C1" "C2" "D1" "D2"