#!/bin/bash

#SBATCH -J spcyl_relax_p75
#SBATCH -e spcyl_rlx_err.log
#SBATCH -o spcyl_rlx_out.log
#SBATCH --mem-per-cpu=2000
#SBATCH -t 5:00:00
#SBATCH -n 1
#SBATCH -p standard

srun ~/spherocyls/exe/cpp_spherocyl_minimize r /scratch/tmarscha/spherocyl/minimization/n512/p750/r3/ 512 0.025 1000000 1000 1 0.1 0.75 2.0 0.5