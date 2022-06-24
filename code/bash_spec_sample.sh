#!/bin/bash
#SBATCH -N 1
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
srun -n 1 -c 64 --cpu_bind=core python double_sample_spec_copy.py
