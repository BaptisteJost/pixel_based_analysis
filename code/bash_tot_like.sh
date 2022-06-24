#!/bin/bash
#SBATCH -N 10
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH --mail-user=jost@apc.in2p3.fr
#SBATCH --mail-type=ALL
#SBATCH -t 00:30:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


#run the application:
conda activate myenv
srun -n 640 -c 1 --cpu_bind=threads python /global/homes/j/jost/these/pixel_based_analysis/code/total_likelihood.py
