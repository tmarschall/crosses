#!/bin/bash

#SBATCH -J spcyl_relax_debug
#SBATCH -e spcyl_rlxdb_err.log
#SBATCH -o spcyl_rlxdb_out.log
#SBATCH --mem-per-cpu=2000
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH -p debug

srun ~/spherocyls/exe/cpp_spherocyl_minimize r /scratch/tmarscha/spherocyl/test/ 512 0.025 1000000 1000 1 0.1 0.8 2.0 0.5