#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -N two_qubit_test

cd $PBS_O_WORKDIR
time python two_qubit_test.py