#!/bin/bash


#SBATCH -J cross_shear_test
#SBATCH -e cross_shear_err.log
#SBATCH -o cross_shear_out.log
#SBATCH --mem-per-cpu=2000
#SBATCH -t 120:00:00
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu

module load cuda

mkdir -p /scratch/$USER/cross/test/
srun ~/crosses/exe/cross_box r /scratch/$USER/cross/test/ 1024 1e-5 0.02 10.0 0.01 0.00002 0.08 0.80 0.5 2.0 0.5
