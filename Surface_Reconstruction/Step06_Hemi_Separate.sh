#!/bin/bash
#SBATCH --job-name=Hemi_Separate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=bme_a10080g
#SBATCH -x bme_gpu[01,02,09]
#SBATCH -t 120:00:00

python -u Step06_Hemi_Separate.py