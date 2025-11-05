#!/bin/bash
#SBATCH --job-name=DK_Structure_Refinement
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --partition=bme_cpu
#SBATCH -t 24:00:00

python -u Step04_DK_Structure_Refinement.py