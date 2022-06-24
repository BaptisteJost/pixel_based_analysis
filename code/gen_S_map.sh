#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q debug
#SBATCH -t 00:30:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


#run the application:
conda activate myenv
srun -n 32 -c 2 --cpu_bind=cores python /global/homes/j/jost/these/pixel_based_analysis/code/generate_SCMB.py
