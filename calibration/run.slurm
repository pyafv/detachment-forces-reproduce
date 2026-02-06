#!/bin/bash -l
#SBATCH --job-name=DP_detach
#SBATCH --time=3:0:0
#SBATCH --account=bcamley1
#SBATCH --mail-type=end,fail,time_limit
#SBATCH --mail-user=wwang138@jh.edu
#SBATCH --ntasks=13
#SBATCH -p shared


### exit on error
set -e

module load openmpi

python --version
mpiexec python dp_detach.py

wait
date
