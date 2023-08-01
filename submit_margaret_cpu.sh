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
#SBATCH --array=0-19

RS_VALUES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
NT_VALUES=(32 64 128 512 1024)

for i in ${NT_VALUES[@]}; do
    srun conda run -n exp_toy python -W ignore evaluate_model_timetest.py -ds "company_employees" -nt $i -i "False" -m "all" -rs ${RS_VALUES[$SLURM_ARRAY_TASK_ID]} -dv "cpu"
done

wait