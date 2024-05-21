#!/bin/bash
#SBATCH --job-name=sftm
#SBATCH --time=1-0
#SBATCH --mem-per-cpu=16GB
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=colins26@stanford.edu
#SBATCH --array=0-65

python -u run_experiment.py -i ${SLURM_ARRAY_TASK_ID}