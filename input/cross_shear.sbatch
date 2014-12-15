#!/bin/bash

#SBATCH -J cross_p10dg1e-5r1
#SBATCH -e cross_shear_err.log
#SBATCH -o cross_shear_out.log
#SBATCH --mem-per-cpu=2000
#SBATCH -t 99:00:00
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu

module load cuda
mkdir -p /scratch/tmarscha/cross/n512/p10/dg1e-5/r1/
srun ~/crosses/exe/cross_box r /scratch/tmarscha/cross/n512/p10/dg1e-5/r1/ 512 1e-5 0.025 100.0 0.01 0.00002 0.09 0.10 0.5 2.0 1.0