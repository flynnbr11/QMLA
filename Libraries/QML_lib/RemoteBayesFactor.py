from __future__ import print_function # so print doesn't show brackets
# Libraries
import numpy as np
import itertools as itr
import os as os
import sys as sys 
import pandas as pd
import warnings
import time as time
import random
from psutil import virtual_memory
import json ## possibly worth a different serialization if pickle is very slow
import pickle
pickle.HIGHEST_PROTOCOL=2
import copy

try:
    from RedisSettings import * 
    enfore_serial = False  
except:
    enforce_serial = True # shouldn't be needed
      
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# Local files
import Evo as evo
import DataBase 
from QML import *
import ModelGeneration
import BayesF
from qinfer import NormalDistribution
from Distrib import MultiVariateNormalDistributionNocov

## Single function call to compute Bayes Factor between models given their IDs

def BayesFactorRemote(model_a_id, model_b_id):
    model_a = modelClassForRemoteBayesFactor(modelID=model_a_id)
    model_b = modelClassForRemoteBayesFactor(modelID=model_b_id)

    times_a = model_a.Times
    times_b = model_b.Times
    
    log_l_a = log_likelihood(model_a, times_b)
    log_l_b = log_likelihood(model_b, times_a)     
    
    bayes_factor = np.exp(log_l_a - log_l_b)
    print("Bayes factor bw", model_a.Name, "/", model_b.Name, "=", bayes_factor)
    return bayes_factor
    
    
    
    
def log_likelihood(model, times):
    updater = model.Updater #this could be what Updater.hypotheticalUpdate is for?
    
    for i in range(len(times)):
    
        exp = get_exp(model, updater.model, [times[i]])
        params_array = np.array([[model.FinalParams[0][0]]])
        datum = updater.model.simulate_experiment(params_array, exp)
        updater.update(datum, exp)

    log_likelihood = updater.log_total_likelihood
    return log_likelihood        

def get_exp(model, gen, time):
    exp = np.empty(len(time), dtype=gen.expparams_dtype)
    exp['t'] = time

    for i in range(1, len(gen.expparams_dtype)):
        col_name = 'w_'+str(i)
        exp[col_name] = model.FinalParams[i-1,0] 
    return exp
    
