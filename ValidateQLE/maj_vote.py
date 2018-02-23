from __future__ import print_function # so print doesn't show brackets
import numpy as np
import itertools as itr

import os as os
import sys as sys 
import pandas as pd
import warnings
import time as time
import random
import pickle
pickle.HIGHEST_PROTOCOL = 2
sys.path.append(os.path.join("..", "Libraries","QML_lib"))
import Evo as evo
import DataBase 
from QMD import QMD #  class moved to QMD in Library
import QML
import ModelGeneration 
import BayesF
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
paulis = ['x', 'y', 'z'] # will be chosen at random. or uncomment below and comment within loop to hard-set

import time as time 
import argparse
parser = argparse.ArgumentParser(description='Pass variables for (I)QLE.')


def get_directory_name_by_time(just_date=False):
    import datetime
    # Directory name based on date and time it was generated 
    # from https://www.saltycrane.com/blog/2008/06/how-to-get-current-date-and-time-in/
    now =  datetime.date.today()
    year = now.strftime("%y")
    month = now.strftime("%b")
    day = now.strftime("%d")
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute
    date = str (str(day)+'_'+str(month)+'_'+str(year) )
    time = str(str(hour)+'_'+str(minute))
    name = str(date+'/'+time+'/')
    if just_date is False:
        return name
    else: 
        return str(date+'/')

### Set up command line arguments to alter script parameters. ###

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
  default=200
)

parser.add_argument(
  '-p', '--num_particles', 
  help='Number of particles to use for the learning process',
  type=int,
  default=300
)

parser.add_argument(
  '-q', '--num_qubits', 
  help='Number of qubits to run tests for.',
  type=int,
  default=2
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
  default=0
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
parser.add_argument(
  '-vary_rt', '--vary_resample_threshold',
  help='Vary resampling threshold for QInfer, i.e. sweep over parameter.',
  type=bool,
  default=False
)
parser.add_argument(
  '-vary_ra', '--vary_resample_a',
  help='Vary resampling threshold for QInfer, i.e. sweep over parameter.',
  type=bool,
  default=False
)
parser.add_argument(
  '-vary_pgh', '--vary_pgh_factor',
  help='Vary resampling threshold for QInfer, i.e. sweep over parameter.',
  type=bool,
  default=False
)

arguments = parser.parse_args()
do_iqle = bool(arguments.iqle)
do_qle = bool(arguments.qle)
num_runs = arguments.num_runs
num_tests = arguments.num_tests
num_qubits = arguments.num_qubits
num_parameters = arguments.num_parameters
num_exp = arguments.num_experiments
num_part = arguments.num_particles
all_plots = bool(arguments.plots)
best_resample_threshold = arguments.resample_threshold
best_resample_a = arguments.resample_a
best_pgh = arguments.pgh_factor
vary_resample_a = arguments.vary_resample_threshold
vary_resample_thresh = arguments.vary_resample_a
vary_pgh_factor = arguments.vary_pgh_factor

#######

plot_time = get_directory_name_by_time(just_date=False) # rather than calling at separate times and causing confusion
save_figs = False
save_data = False

intermediate_plots = all_plots
do_summary_plots = all_plots
store_data = all_plots

global_true_op = ModelGeneration.random_model_name(num_dimensions=num_qubits, num_terms=num_parameters)  # choose a random initial Hamiltonian.

while global_true_op == 'i':
  global_true_op = ModelGeneration.random_model_name(num_dimensions=num_qubits, num_terms=num_parameters)  # choose a random initial Hamiltonian.


global paulis_list
paulis_list = {'i' : np.eye(2), 'x' : evo.sigmax(), 'y' : evo.sigmay(), 'z' : evo.sigmaz()}


#TODO Should we remove these warning?
warnings.filterwarnings("ignore", message='Negative weights occured', category=RuntimeWarning)
warnings.filterwarnings("ignore", message='Extremely small n_ess encountered', category=RuntimeWarning)





##########################################

####### run loops 

#########################################


# This cell runs tests on IQLE and QLE and plots the resulting errors and quadratic losses


if vary_resample_thresh or vary_resample_a or vary_pgh_factor:
    variable_parameter = 'vary'
else:
    variable_parameter = ''


a_options = [best_resample_a]
resample_threshold_options = [best_resample_threshold]
pgh_options = [best_pgh]
    
if vary_resample_thresh : 
    variable_parameter += '_thresh'
    resample_threshold_options = np.arange(0.35, 0.75, 0.1)

if vary_resample_a: 
    variable_parameter += '_a'
    a_options = np.arange(0.85, 0.99, 0.05)

if vary_pgh_factor:
    variable_parameter += '_pgh'
    pgh_options = np.arange(0.6, 1.5, 0.4)

# RUN QMD Loops
start = time.time()
qmd_time = 0
true_param_list=[]
true_op_list = []

initial_op_list = ['x', 'y', 'z', 'xTy', 'xTz', 'yTz']
# initial_op_list = ['xPz', 'xPy', 'xPz', 'yPz']
#initial_op_list = ['x', 'y', 'z']
# initial_op_list = ['xPz']
  
# true_op = global_true_op
true_op = random.choice(initial_op_list)
# true_op = 'xPz'
# true_op = 'x'
global_true_op = true_op

op = DataBase.operator(global_true_op)
n_pars = op.num_constituents
true_params = [np.random.rand() for i in range(n_pars)]
for i in true_params: 
  true_param_list.append(i)

true_op_list.append(true_op)
                
# (Note: not learning between models yet; just learning paramters of true model)


qle_values = [] # qle True does QLE; False does IQLE
if do_qle is True:
    qle_values.append(True)
if do_iqle is True:
    qle_values.append(False)

for qle in qle_values:
    a = time.time() # track time in just QMD
    qmd = QMD(
        initial_op_list=initial_op_list, 
        true_operator=true_op, 
        true_param_list=true_params, 
        num_particles=num_part,
        qle=qle,
        max_num_branches = 0,
        max_num_qubits = 2, 
    )
    # qmd.runAllActiveModelsIQLE(num_exp=num_exp)
#                    qmd.runQMD(num_exp = num_exp, spawn=False, just_given_models=True)
#                    qmd.runAllActiveModelsIQLE(num_exp = num_exp)
    qmd.majorityVoteQMD(num_runs=num_runs, num_exp=num_exp, just_given_models=True)
    b = time.time()
    qmd_time += b-a

    pickle.dump(qmd, open("qmd_class.npy", "wb"))
    del qmd




c=time.time()
    

d=time.time()
end = time.time()

num_qle_types = do_iqle + do_qle
num_exponentiations = num_qle_types * num_part * num_exp*num_tests

print("\n\n\n")
print(num_qubits, "Qubits")
print(num_qle_types, "Types of (I)QLE")
print(num_part, "Particles")
print(num_exp, "Experiments")
print(num_tests, "Tests")
print("Totalling ", num_exponentiations, " calls to ", num_qubits,"-qubit exponentiation function.")
print("QMD-time / num exponentiations = ", qmd_time/num_exponentiations)
print("Total time on QMD : ", qmd_time, "seconds.")
print("Total time : ", end - start, "seconds.")
print("True model: ", global_true_op)
# print("QMD Champion:", champion)









