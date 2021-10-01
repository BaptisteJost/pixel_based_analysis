#!/bin/bash
#SBATCH -N 2
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J S_2k_64
#SBATCH --mail-user=jost@apc.in2p3.fr
#SBATCH --mail-type=ALL
#SBATCH -t 02:00:00


#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


#run the application:
srun -n 64 -c 2 --cpu_bind=cores python /global/homes/j/jost/these/pixel_based_analysis/code/generate_SCMB.py
