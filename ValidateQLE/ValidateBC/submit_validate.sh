#!/bin/bash

#PBS -l nodes=1:ppn=1
#PBS -q short
#PBS -N validate
#PSB -O validate

cd $PBS_O_WORKDIR
time python3 run_qmd_new_lib.py