#!/bin/bash

function wait_for_file() {
    local the_file="$1"

    echo -n "Waiting for ${the_file} "
    LOADING=("/" "-" "\\")
    i=0
    echo -ne "${LOADING[0]}"
    while [[ ! -e "${the_file}" ]]; do
        echo -ne "\b${LOADING[$i]}"
        i=$(( (i + 1) % ${#LOADING[@]}))
        sleep 0.1s
    done
    echo -e "\b\b. Found!"
}

function run_data_generator() {
  # Cleanup
  scancel -u s2423286

  # Start iteration
  echo "Starting master node..." 1>&2
  sbatch ${MY_PATH}/master.cmd
  echo "Starting $WORKER_NUM worker(s)..." 1>&2
  sbatch --nodes=${WORKER_NUM} ${MY_PATH}/workers.cmd ${WORKER_NUM}
  echo ""
}

# Default to three workers
WORKER_NUM=${1:-5}
MY_PATH="$(dirname $(realpath $0))"

# Cleanup old run
rm -rf ${MY_PATH}/logs/*

run_data_generator
sleep 15m

# Loop master-worker initiation until completion
while [[ -e "${MY_PATH}/../supervisor_lock.init" ]]; do
  run_data_generator
  sleep 15m
done

# Stop all
scancel -u s2423286
