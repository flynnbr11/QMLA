#!/bin/bash

#PBS -l nodes=1:ppn=1
#PBS -q short
#PBS -N validate_five_qubits_five_tests_no_plots
# PBS -o validate_five_qubits_five_tests_no_plots

cd $PBS_O_WORKDIR
time python3 run_qmd.py
