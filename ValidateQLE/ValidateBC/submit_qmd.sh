#!/bin/bash

#PBS -l nodes=1:ppn=1
#PBS -q short
#PBS -N four_qubits_20_tests
#PBS -o four_qubits_20_tests

cd $PBS_O_WORKDIR
time python3 run_qmd.py -t=20 -pt=True