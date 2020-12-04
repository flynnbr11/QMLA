#!/bin/bash

# mpirun -np 12 python3 hello_world.py
rm test_out.txt
rm output.log

mpirun -np 6 python3 run_sweep_via_mpi.py > test_out.txt
