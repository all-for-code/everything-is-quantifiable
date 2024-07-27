#!/bin/bash

task_name=test_torch

export PYTHONPATH=../:$PYTHONPATH

nohup \
python ../test_torch.py \
> logs/${task_name}.log 2>&1 &

echo $! > logs/${task_name}.pid

