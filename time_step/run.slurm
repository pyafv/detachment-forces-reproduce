#!/bin/bash -l
#SBATCH --job-name=time_step
#SBATCH --time=2:0:0
#SBATCH --account=bcamley1
#SBATCH --mail-type=end,fail,time_limit
#SBATCH --mail-user=wwang138@jh.edu
#SBATCH --ntasks=480
#SBATCH -p parallel


### exit on error
set -e

module load slurm
module load openmpi

python --version
mpiexec python scan_dt.py --delta 0.0
mpiexec python scan_dt.py --delta 0.45

wait
date
