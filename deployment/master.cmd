#!/bin/bash

#SBATCH --job-name=master
#SBATCH --time=0:15:00
#SBATCH --output=deployment/logs/%A_master-out
#SBATCH --error=deployment/logs/%A_master-err
#SBATCH --signal=TERM@120
#SBATCH --partition=dslcALL

#Figure out where we are located
if [[ -n $SLURM_JOB_ID ]]; then
        MY_PATH="$(scontrol show job $SLURM_JOB_ID | grep 'Command' | cut -d'=' -f2)"
else
        MY_PATH="$(realpath $0)"
fi

MY_PATH="$(dirname ${MY_PATH})"

cd ${MY_PATH}/../
export PYTHONPATH=$PYTHONPATH:${MY_PATH}/../
source ${MY_PATH}/../venv3/bin/activate
srun python -u ${MY_PATH}/../supervise.py --daemon

sleep infinity
