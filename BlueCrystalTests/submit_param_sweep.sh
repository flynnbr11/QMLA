#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -N param_sweep

cd $PBS_O_WORKDIR
time python basic_param_sweep.py