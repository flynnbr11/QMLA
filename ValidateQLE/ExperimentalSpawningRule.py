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
import argparse

parser = argparse.ArgumentParser(description='Pass variables for (I)QLE.')

import warnings
warnings.filterwarnings("ignore")


num_runs=1
num_particles= 5
num_experiments = 5 
num_times_bayes = 'all'
qle=True


import time
start = time.time()

initial_op_list = ['xTi', 'yTi', 'zTi']
#initial_op_list = ['x', 'y', 'z']

num_ops = len(initial_op_list)
for i in range(1):
    print("i=",i)
    true_op = 'xTiPPyTiPPzTiPPxTxPPyTyPPzTz'
    true_params = [np.random.rand()]
    qmd = QMD(
        initial_op_list=initial_op_list, 
        true_operator=true_op, 
        true_param_list=None, 
        num_particles=num_particles,
        num_experiments = num_experiments, 
        num_times_for_bayes_updates = num_times_bayes,
        qle=qle,
        num_probes=5,
        max_num_branches = 0,
        max_num_qubits = 10, 
        parallel = True,
        growth_generator='experimental_qmd'
        #growth_generator='experimental_qmd'
    )
   # qmd.learnModelNameList(model_name_list=['z'], blocking=True, use_rq=False)
    qmd.runRemoteQMD(num_spawns=3)
    
end = time.time()
print("\n\nTime taken:", end-start)

