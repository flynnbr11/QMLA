#!/bin/bash
#PBS -l nodes=1:ppn=1
#PBS -N seven_qubit

cd $PBS_O_WORKDIR
time python seven_qubit.py