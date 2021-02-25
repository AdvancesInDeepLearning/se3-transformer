#!/bin/bash

function run_data_generator() {
  # Cleanup
  scancel -u $(whoami)

  # Start iteration
  echo "Starting master node..." 1>&2
  sbatch ${MY_PATH}/master.cmd
  echo "Starting $WORKER_NUM worker(s)..." 1>&2
  sbatch --nodes=${WORKER_NUM} ${MY_PATH}/workers.cmd ${WORKER_NUM}
  echo ""
}

# Default to three workers
WORKER_NUM=${1:-5}
SLEEP_TIME="15m"
MY_PATH="$(dirname $(realpath $0))"

# Cleanup old run
rm -rf ${MY_PATH}/logs/*

# We run it a single time outside the loop, since we expect the
# Master node to drop a .lock-file once it's started.
# This .lock-file will be removed by the master once everything is done
# to indicate this SLURM script here to terminate.
run_data_generator
sleep $(SLEEP_TIME)

# Loop master-worker initiation until completion
while [[ -e "${MY_PATH}/../supervisor_lock.init" ]]; do
  run_data_generator
  sleep $(SLEEP_TIME)
done

# Stop all
scancel -u $(whoami)
