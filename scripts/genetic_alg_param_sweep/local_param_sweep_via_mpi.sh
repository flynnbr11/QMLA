#!/bin/bash

# mpirun -np 12 python3 hello_world.py
rm test_out.txt

mpirun -np 8 python3 run_sweep_via_mpi.py > test_out.txt
