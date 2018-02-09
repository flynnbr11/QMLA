import time 
import hamiltonian_exponentiation as h
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np
import sys, os
sys.path.append(os.path.join("..", "..", "Libraries","QML_lib"))
from ModelGeneration import random_model_name
from DataBase import operator


import argparse
parser = argparse.ArgumentParser(description='Evaluate timings for exp_ham function.')


parser.add_argument(
  '-p', '--plots', 
  help="Save plots: True or False",
  type=bool,
  default=False
)

parser.add_argument(
  '-m', '--machine_type', 
  help="Describe machine. i.e. Local, Cluster_Backend, Cluster_Frontend, etc.",
  type=str,
  default='Local'
)

parser.add_argument(
  '-min',
  help="Minimum qubit number to test.",
  type=int,
  default=1
)
parser.add_argument(
  '-max',
  help="Minimum qubit number to test.",
  type=int,
  default=10
)
parser.add_argument(
  '-tests',
  help="Number of tests to run.",
  type=int,
  default=10
)




arguments = parser.parse_args()
machine_type = arguments.machine_type
save_plots = arguments.plots
min_qubit = arguments.min
max_qubit = arguments.max
num_tests = arguments.tests
qubits = range(min_qubit, max_qubit+1)

times = {}
min_times_dict={}
avg_times_dict={}
minimum_times = []
average_times=[]

for num_qubits in range(min_qubit, 1+max_qubit):
    times[num_qubits] = 0

    
for num_qubits in qubits:
    min_time = 1
    for i in range(num_tests):
        mod = random_model_name(num_dimensions=num_qubits)
        op = operator(mod)
        hamiltonian = op.matrix
        before = time.time()
        store = h.exp_ham(hamiltonian, 1)
        after = time.time()
        if after-before < min_time :
            min_time = after - before
        times[num_qubits] += after - before
    minimum_times.append(min_time)


for q in qubits: 
    average_times.append(times[q]/num_tests)
    avg_times_dict[q] = average_times[q-1]
    min_times_dict[q] = minimum_times[q-1]

# Plots

description = str('_times_'+machine_type+'_machine_upto_'+str(max_qubit)+'_qubits')
title_description = str('Timings of Custom Exponentiation function (' + str(num_tests) + ' tests; ' +  machine_type+ ')')

if save_plots:
  # Average timings
  method='Average'
  plt.clf()
  x=qubits
  y=average_times
  plt.ylabel('Time')
  plt.xlabel('Number of Qubits')
  plt.title(str(method)+str(title_description))
  plt.axhline(y=1, label='1 sec', color='red')
  plt.legend()
  plt.semilogy(x,y, marker='o')
  plt.savefig('Avergage'+description)


  ## Minumum timings
  method='Minimum'
  plt.clf()
  x=qubits
  y=minimum_times
  plt.ylabel('Time')
  plt.xlabel('Number of Qubits')
  plt.title(method+title_description)
  plt.axhline(y=1, label='1 sec', color='red')
  plt.legend()
  plt.semilogy(x,y, marker='o')
  plt.savefig('Minimum'+description)

  np.save("times_"+machine_type+"_"+str(method)+".npy", min_times_dict)
  np.save("avg_times_"+machine_type+"_"+str(method)+".npy", avg_times_dict)    


    
