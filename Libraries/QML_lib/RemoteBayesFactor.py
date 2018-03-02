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

def BayesFactorRemote(model_a_id, model_b_id, num_times_to_use = 'all', check_db=False, trueModel=None):
    if check_db: # built in to only compute once and always return the stored value.
        if pair_id in bayes_factors_db.keys():
            bayes_factor = bayes_factors_db.get(pair_id)
            if model_a_id < model_b_id:
                return bayes_factor
            else:
                return (1.0/bayes_factor)
    else:
        
        model_a = modelClassForRemoteBayesFactor(modelID=model_a_id)
        model_b = modelClassForRemoteBayesFactor(modelID=model_b_id)


        if num_times_to_use == 'all' or len(model_a.Times) < num_times_to_use:
            times_a = model_a.Times
            times_b = model_b.Times
        else:
            times_a = model_a.Times[num_times_to_use:]
            times_b = model_b.Times[num_times_to_use:]
        
        log_l_a = log_likelihood(model_a, times_b)
        log_l_b = log_likelihood(model_b, times_a)     

        bayes_factor = np.exp(log_l_a - log_l_b)
        print("Bayes factor bw", model_a.Name, "/", model_b.Name, "=", bayes_factor)
        
        
        pair_id = DataBase.unique_model_pair_identifier(model_a_id, model_b_id)
        if float(model_a_id) < float(model_b_id): # so that BF in db always refers to (a/b), not (b/a). 
            bayes_factors_db.set(pair_id, bayes_factor)
        else:
            bayes_factors_db.set(pair_id, (1.0/bayes_factor))
        
        if trueModel is not None:
            if (trueModel!=model_a.Name and trueModel!=model_b.Name):
                print("Neither model correct")
            elif bayes_factor > 1 and trueModel == model_a.Name:
                print("Bayes Correct")
            elif bayes_factor < 1 and trueModel == model_b.Name:
                print("Bayes Correct")
            else:
                print("Bayes Incorrect")
        
        
        return bayes_factor
    
    
    
    
def log_likelihood(model, times):
    updater = model.Updater #this could be what Updater.hypotheticalUpdate is for?
    print("Loglikelihoods for", model.Name)
#    print(model.Name, "has final params", model.FinalParams)
#    print("New times considered:", times)
    print("Before:", updater.log_total_likelihood)
    sum_data = 0
    for i in range(len(times)):
        exp = get_exp(model, [times[i]])
#        params_array = np.array([[model.FinalParams[0][0]]])
        params_array = np.array([[model.TrueParams[0]]]) # TODO this will cause an error for multiple parameters
        datum = updater.model.simulate_experiment(params_array, exp, repeat=1)
        # print("\ti=",i, "Time=", times[i], "Exp=", exp, "Datum=", datum)
        sum_data+=datum       
        updater.update(datum, exp)
       # print("Update", i, "Loglikelihood=", updater.log_total_likelihood)

    print("After:", updater.log_total_likelihood)
    fraction_ones = (float(sum_data)/len(times))*100
    print("Sum of data:", sum_data, " 1s out of ", len(times),"=", fraction_ones, "%")
    log_likelihood = updater.log_total_likelihood
    
    return log_likelihood        

def get_exp(model, time):
    gen = model.Updater.model # or gen=model.GenSimModel
    exp = np.empty(len(time), dtype=gen.expparams_dtype)
    exp['t'] = time

    for i in range(1, len(gen.expparams_dtype)):
        col_name = 'w_'+str(i)
        exp[col_name] = model.FinalParams[i-1,0] 
    return exp

"""
OLD but I think equivalent to above?

def get_exp(model, gen, time):
    exp = np.empty(len(time), dtype=gen.expparams_dtype)
    exp['t'] = time

    for i in range(1, len(gen.expparams_dtype)):
        col_name = 'w_'+str(i)
        exp[col_name] = model.FinalParams[i-1,0] 
    return exp
"""



