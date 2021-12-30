#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
    $(dirname "$0")/benchmark.py $CONFIG $CHECKPOINT --launcher pytorch 
