#!/bin/bash

#SBATCH -J cross_resz_test
#SBATCH -e cross_resz_err.log
#SBATCH -o cross_resz_out.log
#SBATCH --mem-per-cpu=2000
#SBATCH -t 120:00:00
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu

module load cuda

mkdir -p /scratch/$USER/cross/test/resize/
srun ~/crosses/exe/cross_box_resize r /scratch/tmarscha/cross/test/resize/ 1024 -1e-6 0.02 0.85 0.001 0.00001 0.08 0.3 0.5 2.0 1.0 1.0
