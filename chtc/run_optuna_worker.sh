#!/bin/bash
# run_optuna_worker.sh

DATASET=$1
MODEL=$2

sleep $((RANDOM % 60)) # random 1-60 seconds

tar -xzvf structureddata.tar.gz
cd structureddata
python hps.py $DATASET $MODEL /staging/groups/cs_geodes/sd/hps
