#!/bin/bash

#SBATCH -J spherocyl_shear
#SBATCH -e spherocyl_shear_err.log
#SBATCH -o spherocyl_shear_out.log
#SBATCH --mem-per-cpu=2000
#SBATCH -t 96:00:00
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -p standard

module load cuda
srun ~/spherocyls/exe/spherocyl_box r /scratch/tmarscha/spherocyl/test/ 512 1e-4 0.025 5.0 0.01 0.00001 0.1 0.7 2.0 0.5