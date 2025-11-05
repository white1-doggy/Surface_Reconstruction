#!/bin/bash
#SBATCH --job-name=Tissue_2_Hemi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --partition=bme_cpu
#SBATCH -t 24:00:00

python -u Step07_Tissue_2_Hemi.py