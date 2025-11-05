#!/bin/bash
#SBATCH --job-name=Intensity_2_Edge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=16
#SBATCH --partition=bme_cpu
#SBATCH -t 12:00:00

python -u Step00_Intensity_2_Edge.py