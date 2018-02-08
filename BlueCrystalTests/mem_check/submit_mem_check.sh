#!/bin/bash 
#PBS -l nodes=1:ppn=1
#PBS -N mem_check

cd $PBS_WORKDIR
time python3 check_mem_usage.py