import os as os
import warnings
import numpy as np
import itertools as itr
import matplotlib.pyplot as plt
import sys as sys
import pandas as pd
import warnings
import time as time
import logging
import random
import pickle
pickle.HIGHEST_PROTOCOL = 4

p = os.path.abspath(os.path.realpath(__file__))
elements = p.split('/')[:-2]
qmla_root = os.path.abspath('/'.join(elements))
sys.path.append(qmla_root)

import qmla
from qmla import construct_models

# from expm import expm
from qmla.shared_functionality.matrix_exponential import expm
from scipy.linalg import expm as lexpm


num_iterations = 100
t_expm = 0
t_lexpm = 0 

for i in range(num_iterations):

    model_params = {
        'FH-hopping-sum_up_1h2_1h3_2h4_3h4_d4': np.random.uniform(0,1), 
        'FH-onsite-sum_1_2_3_4_d4': np.random.uniform(0,1), 
        'FH-hopping-sum_down_1h2_1h3_2h4_3h4_d4': np.random.uniform(0,1)
    }
    model = sum([
        model_params[term] * qmla.construct_models.compute(term) for term in model_params
    ])

    t = np.random.uniform(0,100)



    start_expm = time.time()
    u = expm(-1j*model*t)
    t_expm += time.time() - start_expm

    start_lexpm = time.time()
    u = lexpm(-1j*model*t)
    t_lexpm += time.time() - start_expm

print("Total times taken:\n \texpm={} \n\tlexpm={} \n\tSpeedup={}".format(
    np.round(t_expm, 2), np.round(t_lexpm, 2), np.round(t_lexpm/t_expm, 2)
))


