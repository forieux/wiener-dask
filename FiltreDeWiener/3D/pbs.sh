#!/bin/bash

#PBS -l walltime=00:20:00
#PBS -l select=8:ncpus=4:mem=32gb
#PBS -M fabioeid.morooka@l2s.centralesupelec.fr
#PBS -m e
#PBS -N dask_seq
#PBS -j oe
#PBS -P gpi

# Load necessary modules
module load anaconda2/2019.03

# Activate anaconda environment
source activate myenv

# Move to directory where the job was submitted
cd $PBS_O_WORKDIR

# Run python script
python daskPure.py

