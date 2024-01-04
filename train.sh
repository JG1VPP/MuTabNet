#!/usr/bin/env bash

if [ $# -lt 3 ]
then
    echo "Usage: bash $0 CONFIG WORK_DIR GPUS"
    exit
fi

BIN=${BIN:-python3}
CONFIG=$1
WORK_DIR=$2
GPUS=$3

PORT=${PORT:-29500}
SCRIPT=$(dirname $0)/train.py

if [ ${GPUS} == 1 ]; then
    $BIN $SCRIPT $CONFIG --work-dir=${WORK_DIR}
else
    $BIN -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT $SCRIPT $CONFIG --work-dir=${WORK_DIR} --launcher pytorch
fi
