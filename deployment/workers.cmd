#!/bin/bash

#SBATCH --job-name=workers
#SBATCH --time=0:15:00
#SBATCH --output=deployment/logs/%A_%a_worker-out
#SBATCH --error=deployment/logs/%A_%a_worker-err
#SBATCH --signal=TERM@120
#SBATCH --partition=dslcALL

#Figure out where we are located
if [[ -n $SLURM_JOB_ID ]]; then
        MY_PATH="$(scontrol show job $SLURM_JOB_ID | grep 'Command' | cut -d'=' -f2)"
else
        MY_PATH="$(realpath $0)"
fi

MY_PATH="$(dirname ${MY_PATH})"

#On DSLC the venv is not automatically enabled.
#Thus, we manually navigate to where we need to be and acivate it manually
cd ${MY_PATH}/../
source ${MY_PATH}/../venv3/bin/activate

#Tell slurm how many worker nodes shall be used
srun --nodes=$1 python -u generate.py

sleep infinity
