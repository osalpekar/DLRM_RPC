#!/bin/bash

#SBATCH --job-name=dlrm_rpc

#SBATCH --partition=q3

#SBATCH --nodes=3

#SBATCH --ntasks-per-node=1

#SBATCH --gpus-per-node=8

# Steps to run:
# 1. Activate environment
# 2. Run sbatch ./sbatch_launch.sh <script_name>
# e.g. sbatch ./sbatch_launch.sh ./train_sbatch.sh
if [ -e $1 ];
then
    echo "Executing $1 with slurm batch."
else
    echo "$1 does not exist."
fi

srun --label $1
