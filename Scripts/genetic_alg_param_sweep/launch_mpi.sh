#!/bin/bash

# mpirun -np 12 python3 hello_world.py
mpirun -np 6 python3 sweep_parallel.py > test_out.txt
