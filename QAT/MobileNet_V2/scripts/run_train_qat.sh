#!/bin/bash

task_name=train_torch_qat_fix_mse_frbn_v2

export PYTHONPATH=../:$PYTHONPATH

nohup \
python ../train_torch_qat.py \
> logs/${task_name}.log 2>&1 &

echo $! > logs/${task_name}.pid

