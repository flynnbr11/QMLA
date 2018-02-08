#!/bin/bash

declare -a experiment_numbers=(10 14 19)
declare -a particle_numbers=(12 13)
declare -a qubit_numbers=(2 1)




for qubits in "${qubit_numbers[@]}" 
do
  for part in "${particle_numbers[@]}"
  do
    for exp in "${experiment_numbers[@]}"
    do
      python3 run_qmd.py -e=$exp -p=$part -q=$qubits -pt=False
    done
  done
done  
