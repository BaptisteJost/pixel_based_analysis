#!/bin/bash
#SBATCH -N 1
#SBATCH -C haswell
#SBATCH -q regular
#SBATCH -J prior_precision_testjaccosmo
#SBATCH --mail-user=jost@apc.in2p3.fr
#SBATCH --mail-type=ALL
#SBATCH -t 01:15:00

#OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread


#run the application:
srun -n 4 -c 16 --cpu_bind=cores python /global/homes/j/jost/these/pixel_based_analysis/code/prior_precison_gridding.py
srun -n 1 -c 64 --cpu_bind=cores python /global/homes/j/jost/these/pixel_based_analysis/code/graph_prior_precision.py
