#!/bin/bash
#
#SBATCH --job-name=test_job
#SBATCH --output=./results/slurm/res_test_job_%A_%a.txt
#
#SBATCH --ntasks=1
#SBATCH --partition=normal,parietal
#SBATCH --error ./results/slurm/error_%A_%a.out
#SBATCH --cpus-per-task=64
# 
#SBATCH --time 100:00:00
#SBatch --exclusive 
#SBATCH --array=0-1

RS_VALUES=(1 2)
NT_VALUES=(128)
DATA_NAME=("michelin")
INCLUDE_NUMERIC=("True")
METHOD_NAME=("fasttext_resnet")
DEVICE=("cpu")

for i in ${NT_VALUES[@]}; do
    srun conda run -n exp_toy python -W ignore evaluate_model.py -ds ${DATA_NAME[@]} -nt $i -i ${INCLUDE_NUMERIC[@]} -m ${METHOD_NAME[@]} -rs ${RS_VALUES[$SLURM_ARRAY_TASK_ID]} -dv ${DEVICE[@]}
done

wait