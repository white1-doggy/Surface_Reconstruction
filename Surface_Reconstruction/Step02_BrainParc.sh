#!/bin/bash
#SBATCH --job-name=BrainParc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=bme_gpu
#SBATCH -x bme_gpu[01,02,09]
#SBATCH -t 48:00:00

python -u Step02_BrainParc.py