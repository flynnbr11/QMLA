#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -N eight_qubit

cd $PBS_O_WORKDIR
time python eight_qubit.py