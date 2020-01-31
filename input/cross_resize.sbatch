#!/bin/bash

#SBATCH -J cross_resz_p65dg1e-9r3
#SBATCH -e cross_resz_err.log
#SBATCH -o cross_resz_out.log
#SBATCH --mem-per-cpu=2000
#SBATCH -t 120:00:00
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p gpu

module load cuda

mkdir -p /scratch/tmarscha/cross/resize/n512/p65/dr1e-9/r3/
srun ~/crosses/exe/cross_box_resize r /scratch/tmarscha/cross/resize/n512/p65/dr1e-9/r3/ 512 -1e-9 0.025 0.65 0.001 0.00001 0.08 0.3 0.5 2.5 2.5