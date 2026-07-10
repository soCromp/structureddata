#!/bin/bash
# run_optuna_worker.sh

DATASET=$1
MODEL=$2

tar -xzvf structureddata.tar.gz
cd structureddata
python hps.py $DATASET $MODEL /staging/groups/cs_geodes/sd/hps
