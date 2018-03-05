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
from pympler import asizeof
import matplotlib.pyplot as plt
paulis = ['x', 'y', 'z'] # will be chosen at random. or uncomment below and comment within loop to hard-set

import time as time 
import argparse
parser = argparse.ArgumentParser(description='Pass variables for (I)QLE.')


initial_op_list = ['x', 'y', 'z', 'xTx', 'xPy', 'xTz', 'xTyPPzTy']
num_ops = len(initial_op_list)
num_particles = 400
num_experiments = 200
qle=True

for i in range(num_ops):
    print("i=",i)

    #initial_op_list = ['x', 'y', 'z']
    true_op=initial_op_list[i%num_ops]
    #true_op = random.choice(initial_op_list)
    global_true_op = true_op

    op = DataBase.operator(global_true_op)
    n_pars = op.num_constituents
    true_params = [np.random.rand() for i in range(n_pars)]
    #true_params = [0.512]

    qmd = QMD(
        initial_op_list=initial_op_list, 
        true_operator=true_op, 
        true_param_list=true_params, 
        num_particles=num_particles,
        num_experiments = num_experiments, 
        qle=qle,
        num_probes=5,
        max_num_branches = 0,
        max_num_qubits = 2, 
        parallel = True
    )
    qmd.learnModelNameList(model_name_list=initial_op_list, blocking=True, use_rq=False)
    ids=DataBase.active_model_ids_by_branch_id(qmd.db, 0)
    qmd.remoteBayesFromIDList(ids, remote=False)
    # qmd.processRemoteBayesFactors()
    flushdatabases()
    del qmd
    print("\n\n")
