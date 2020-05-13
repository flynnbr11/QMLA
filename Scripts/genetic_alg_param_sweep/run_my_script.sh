#!/bin/bash

let NUM_WORKERS="$PBS_NUM_NODES * $PBS_NUM_PPN"

mpiexec -n $NUM_WORKERS python3 sweep_parallel.py  > out.txt