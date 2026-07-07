#!/bin/bash
# run_optuna_worker.sh

DATASET=$1
MODEL=$2

tar -xzvf structureddata.tar.gz
cd structureddata
python tune_hps.py $DATASET $MODEL /staging/c/cromp/sd/hps
