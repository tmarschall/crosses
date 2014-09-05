#!/bin/bash

#SBATCH -J cross_shear
#SBATCH -e cross_shear_err.log
#SBATCH -o cross_shear_out.log
#SBATCH --mem-per-cpu=2000
#SBATCH -t 5:00:00
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p graphics

module load cuda
srun ~/crosses/exe/cross_box r /scratch/tmarscha/cross/test/ 512 1e-4 0.025 5.0 0.01 0.00001 0.1 0.4 0.5 2.0 1.0