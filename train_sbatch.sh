#!/bin/bash

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
#export NTASKS=${SLURM_NTASKS}
export PROCID=${SLURM_PROCID}

python driver.py \
    --use-gpu \
    --num-nodes 3 \
    --node-rank ${SLURM_PROCID} \
    --num-trainers 8 \
    --master-addr ${MASTER_ADDR} \
    &> "out${SLURM_PROCID}"
