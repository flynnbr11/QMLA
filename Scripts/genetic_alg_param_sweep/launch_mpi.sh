#!/bin/bash

# mpirun -np 12 python3 hello_world.py
mpirun -np 4 python3 sweep_parallel.py > test_out.txt
