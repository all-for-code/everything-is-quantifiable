#!/bin/bash

task_name=train_torch

export PYTHONPATH=../:$PYTHONPATH

nohup \
python ../train_torch.py \
> logs/${task_name}.log 2>&1 &

echo $! > logs/${task_name}.pid

