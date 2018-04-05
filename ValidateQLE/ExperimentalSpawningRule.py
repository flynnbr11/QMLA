from __future__ import print_function # so print doesn't show brackets
import os as os

import argparse
parser = argparse.ArgumentParser(description='Pass variables for (I)QLE.')

import warnings
warnings.filterwarnings("ignore")

## Parse input variables to use in QMD 
parser.add_argument(
  '-r', '--num_runs', 
  help="Number of runs to perform majority voting.",
  type=int,
  default=1
)
parser.add_argument(
  '-t', '--num_tests', 
  help="Number of complete tests to average over.",
  type=int,
  default=1
)

parser.add_argument(
  '-e', '--num_experiments', 
  help='Number of experiments to use for the learning process',
  type=int,
  default=5
)
parser.add_argument(
  '-p', '--num_particles', 
  help='Number of particles to use for the learning process',
  type=int,
  default=10
)
parser.add_argument(
  '-bt', '--bayes_times', 
  help='Number of times to consider in Bayes function.',
  type=int,
  default=2
)

parser.add_argument(
  '-q', '--num_qubits', 
  help='Number of qubits to run tests for.',
  type=int,
  default=2
)
parser.add_argument(
  '-rq', '--use_rq', 
  help='Bool whether to use RQ for parallel or not.',
  type=int,
  default=1
)
parser.add_argument(
  '-pm', '--num_parameters', 
  help='Number of parameters to run tests for.',
  type=int,
  default=1
)


parser.add_argument(
  '-qle',
  help='True to perform QLE, False otherwise.',
  type=int,
  default=1
)
parser.add_argument(
  '-iqle',
  help='True to perform IQLE, False otherwise.',
  type=int,
  default=1
)


parser.add_argument(
  '-pt', '--plots',
  help='True: do generate all plots for this script; False: do not.',
  type=int,
  default=0
)
parser.add_argument(
  '-rt', '--resample_threshold',
  help='Resampling threshold for QInfer.',
  type=float,
  default=0.6
)
parser.add_argument(
  '-ra', '--resample_a',
  help='Resampling a for QInfer.',
  type=float,
  default=0.9
)
parser.add_argument(
  '-pgh', '--pgh_factor',
  help='Resampling threshold for QInfer.',
  type=float,
  default=1.0
)


arguments = parser.parse_args()
do_iqle = bool(arguments.iqle)
do_qle = bool(arguments.qle)
use_rq = bool(arguments.use_rq)
num_runs = arguments.num_runs
num_tests = arguments.num_tests
num_qubits = arguments.num_qubits
num_parameters = arguments.num_parameters
num_experiments = arguments.num_experiments
num_particles = arguments.num_particles
num_times_bayes = arguments.bayes_times
if num_times_bayes > num_experiments:
    num_times_bayes = num_experiments-1
all_plots = bool(arguments.plots)
best_resample_threshold = arguments.resample_threshold
best_resample_a = arguments.resample_a
best_pgh = arguments.pgh_factor


print("Setting environment variable USE_RQ to", use_rq)
os.environ["USE_RQ"] = str(use_rq)


import numpy as np
import itertools as itr

import sys as sys 
import pandas as pd
import warnings
import time as time
import random
import pickle
pickle.HIGHEST_PROTOCOL = 2
sys.path.append(os.path.join("..", "Libraries","QML_lib"))
print("now import from redis settings")
from RedisSettings import *
import Evo as evo
import DataBase 
from QMD import QMD #  class moved to QMD in Library
import QML
import ModelGeneration 
import BayesF
import matplotlib.pyplot as plt
#from pympler import asizeof
import matplotlib.pyplot as plt
paulis = ['x', 'y', 'z'] # will be chosen at random. or uncomment below and comment within loop to hard-set

import time as time 


###  START QMD ###

print("num times to use:", num_times_bayes)
qle=do_iqle
pickle_result_db = True

import time
start = time.time()

initial_op_list = ['xTi', 'yTi', 'zTi']
#initial_op_list = ['x', 'y', 'z']

num_ops = len(initial_op_list)
for i in range(num_runs):
    print("\ni=",i)
    print(num_particles, "Paricles for ", num_experiments, "Experiments:")
    true_op = 'xTiPPyTiPPzTiPPxTxPPyTyPPzTz'
#    true_params = [np.random.rand()]
    #true_params = [0.19, 0.21, 0.8, 0.22, 0.20, 0.27]
    true_params = [0.25, 0.21, 0.28, 0.22, 0.23, 0.27]
    qmd = QMD(
        initial_op_list=initial_op_list, 
        true_operator=true_op, 
        true_param_list=true_params, 
        num_particles=num_particles,
        num_experiments = num_experiments, 
        num_times_for_bayes_updates = num_times_bayes,
        qle=qle,
        num_probes=5,
        gaussian=True, 
        max_num_branches = 0,
        max_num_qubits = 10, 
        parallel = True,
        use_rq = use_rq,
        use_exp_custom=False, 
        compare_linalg_exp_tol=None,
        growth_generator='ising_non_transverse'
        #growth_generator='hyperfine_like'
    )
   # qmd.learnModelNameList(model_name_list=['z'], blocking=True, use_rq=False)
    qmd.runRemoteQMD(num_spawns=3)
    
    if pickle_result_db:
        print("QMD complete. Pickling result.")
        pickle.dump(qmd, open("qmd_class.p", "wb"), protocol=2)
    
    qmd.plotVolumes(save_to_file='volumes_all_models.png')
    qmd.plotVolumes(branch_champions=True, save_to_file='volumes_branch_champs.png')
    
    
end = time.time()
print("\n\nTime taken:", end-start)

