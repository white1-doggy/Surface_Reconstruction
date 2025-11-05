#!/bin/bash
#SBATCH --job-name=PreSurf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --partition=bme_cpu
#SBATCH -t 24:00:00

python -u Step08_PreSurf.py