from __future__ import print_function # so print doesn't show brackets
import os as os

import warnings
#warnings.filterwarnings("ignore")

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

## Parse input variables to use in QMD 
import GlobalVariables
global_variables = GlobalVariables.parse_cmd_line_args(sys.argv[1:])

import RedisSettings as rds
import Evo as evo
import DataBase 
import ExperimentalDataFunctions as expdt
from QMD import QMD #  class moved to QMD in Library
import QML
import ModelGeneration 
import matplotlib.pyplot as plt
#from pympler import asizeof
import matplotlib.pyplot as plt
import time as time 

###  START QMD ###


import time
start = time.time()

"""
Set up and functions. 
"""

def time_seconds():
    import datetime
    now =  datetime.date.today()
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute
    second = datetime.datetime.now().second
    time = str(str(hour)+':'+str(minute)+':'+str(second))
    return time

def log_print(to_print_list, log_file):
    identifier = str(str(time_seconds()) +" [EXP]")
    if type(to_print_list)!=list:
        to_print_list = list(to_print_list)

    print_strings = [str(s) for s in to_print_list]
    to_print = " ".join(print_strings)
    with open(log_file, 'a') as write_log_file:
        print(identifier, str(to_print), file=write_log_file, flush=True)

log_file = global_variables.log_file
qle = global_variables.do_qle # True for QLE, False for IQLE


num_probes = 40
experimental_probe_dict = expdt.experimental_NVcentre_ising_probes(
    num_probes=num_probes
)

experimental_measurements_dict = expdt.experimentalMeasurementDict(
    directory="NV05_HahnPeaks_expdataset"
)

for k in list(experimental_measurements_dict.keys()):
    # Convert nanoseconds in exp data to milliseconds
    t_new = k/1000
    experimental_measurements_dict[t_new] = experimental_measurements_dict.pop(k)



#print("experimental measurements:", experimental_measurements_dict)

#print("exp probes:", experimental_probe_dict)
use_experimental_measurements = False


initial_op_list = ['xTi', 'yTi', 'zTi']
# true_op = 'xTiPPyTiPPzTiPPxTxPPyTyPPzTz'
true_op = 'xTi'
# true_params = [0.25, 0.21, 0.28, 0.22, 0.23, 0.27]

true_params = []

for i in range(3):
    true_params.append(random.uniform(-1,2))
for i in range(3):
    true_params.append(random.uniform(2,3))


true_params = [0.321]

num_ops = len(initial_op_list)

    
log_print(["\n QMD id", global_variables.qmd_id, " on host ",
    global_variables.host_name, "and port", global_variables.port_number,
    "has seed", rds.get_seed(global_variables.host_name,
    global_variables.port_number, global_variables.qmd_id,
    print_status=True),"\n", global_variables.num_particles,
    " particles for", global_variables.num_experiments, 
    "experiments and ", global_variables.num_times_bayes,
    "bayes updates\n Gaussian=", global_variables.gaussian, 
    "\n RQ=", global_variables.use_rq, "RQ log:",
     global_variables.log_file, "\n Bayes CSV:",
     global_variables.cumulative_csv], log_file
 )

""" 
Launch and run QMD
"""

qmd = QMD(
    initial_op_list=initial_op_list, 
    true_operator=true_op, 
    true_param_list=true_params, 
    num_particles=global_variables.num_particles,
    num_experiments = global_variables.num_experiments, 
    num_times_for_bayes_updates = global_variables.num_times_bayes,
    qle=qle,
    resample_threshold = global_variables.resample_threshold,
    resampler_a = global_variables.resample_a, 
    pgh_prefactor = global_variables.pgh_factor,
    num_probes=num_probes,
    probe_dict = experimental_probe_dict, 
    gaussian = global_variables.gaussian, 
    use_experimental_data = global_variables.use_experimental_data,
    experimental_measurements = experimental_measurements_dict,
    max_num_branches = 0,
    max_num_qubits = 10, 
    parallel = True,
    use_rq = global_variables.use_rq,
    use_exp_custom=False, 
    compare_linalg_exp_tol=None,
    #growth_generator='ising_non_transverse'
#    growth_generator='two_qubit_ising_rotation_hyperfine',
#    growth_generator='two_qubit_ising_rotation_hyperfine',
    growth_generator='qhl_TEST',
    q_id = global_variables.qmd_id,
    host_name = global_variables.host_name,
    port_number = global_variables.port_number,
    rq_timeout = global_variables.rq_timeout,
    log_file = global_variables.log_file
)
# qmd.runRemoteQMD(num_spawns=3)


qmd.runQHLTest()

qmd.delete_unpicklable_attributes()

with open('test_qhl_output.p', "wb") as pkl_file:
    pickle.dump(qmd, pkl_file, protocol=2)




