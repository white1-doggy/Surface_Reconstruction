#!/bin/bash
#SBATCH --job-name=Hippo_Correction
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --partition=bme_cpu
#SBATCH -t 24:00:00

module load compiler/gcc/7.3.1
python -u Step03_Hippo_Correction.py