#!/bin/bash
#PBS -l nodes=1:ppn=8
#PBS -N nine_qubit

cd $PBS_O_WORKDIR
time python nine_qubit.py