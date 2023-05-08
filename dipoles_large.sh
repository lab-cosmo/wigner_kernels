#!/bin/bash
#SBATCH --chdir /home/bigi/wigner_kernels_official/wigner_kernels/
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 20
#SBATCH --gres gpu:1
#SBATCH --mem 80G
#SBATCH --time 5:00:00
export OMP_NUM_THREADS=20
echo STARTING AT `date`
python -u dipoles_large.py $1 $2
echo FINISHED at `date`
