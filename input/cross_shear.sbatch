#!/bin/bash

#SBATCH -J cross_p70dg1e-4r4
#SBATCH -e cross_shear_err.log
#SBATCH -o cross_shear_out.log
#SBATCH --mem-per-cpu=2000
#SBATCH -t 99:00:00
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu

module load cuda

mkdir -p /scratch/tmarscha/cross/n512/p70/dg1e-4/r4/
srun ~/crosses/exe/cross_box r /scratch/tmarscha/cross/n512/p70/dg1e-4/r4/ 512 1e-4 0.025 120.0 0.01 0.00002 0.09 0.70 0.5 2.0 2.0