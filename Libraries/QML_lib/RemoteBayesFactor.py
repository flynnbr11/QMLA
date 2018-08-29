from __future__ import print_function # so print doesn't show brackets
# Libraries
import numpy as np
import itertools as itr
import os as os
import sys as sys 
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import time as time
import random
from psutil import virtual_memory
import json ## possibly worth a different serialization if pickle is very slow
import pickle
pickle.HIGHEST_PROTOCOL=2
import copy

try:
    import RedisSettings as rds
    import redis
    enforce_serial = False  
except:
    enforce_serial = True # shouldn't be needed
      
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# Local files
import Evo as evo
import DataBase 
import QML
import ModelGeneration
#import BayesF
from qinfer import NormalDistribution
from Distrib import MultiVariateNormalDistributionNocov

def time_seconds():
    import datetime
    now =  datetime.date.today()
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute
    second = datetime.datetime.now().second
    time = str(str(hour)+':'+str(minute)+':'+str(second))
    
    return time

## Single function call to compute Bayes Factor between models given their IDs

def BayesFactorRemote(model_a_id, model_b_id, branchID=None, 
    interbranch=False, num_times_to_use = 'all', check_db=False, 
    trueModel=None, bayes_threshold=1, host_name='localhost', port_number=6379, 
    qid=0, log_file='rq_output.log'
):
    """
    This is a standalone function to compute Bayes factors without knowledge 
    of full QMD program. QMD info is unpickled from a redis databse, containing
    learned_model information, i.e. final parameters etc. 
    Given model ids correspond to model names in the database, which are combined 
    with the final learned parameters to reconstruct model classes of 
    complete learned models. 
    From these we extract log likelihoods to compute Bayes factors. 
    
    """

    write_log_file = open(log_file, 'a')
    def log_print(to_print_list):
        identifier = str(str(time_seconds()) + " [RQ Bayes "
            +str(model_a_id)+"/"+str(model_b_id)+"]"
        )
        if type(to_print_list)!=list:
            to_print_list = list(to_print_list)
        print_strings = [str(s) for s in to_print_list]
        to_print = " ".join(print_strings)
        with open(log_file, 'a') as write_log_file:
            print(identifier, str(to_print), file=write_log_file, flush=True)

    time_start = time.time()
    rds_dbs = rds.databases_from_qmd_id(host_name, port_number, qid)
    qmd_info_db = rds_dbs['qmd_info_db'] 
    learned_models_info = rds_dbs['learned_models_info']
    learned_models_ids = rds_dbs['learned_models_ids']
    bayes_factors_db = rds_dbs['bayes_factors_db']
    bayes_factors_winners_db = rds_dbs['bayes_factors_winners_db']
    active_branches_learning_models = rds_dbs['active_branches_learning_models']
    active_branches_bayes = rds_dbs['active_branches_bayes']
    active_interbranch_bayes = rds_dbs['active_interbranch_bayes']
    
    if check_db: # built in to only compute once and always return the stored value.
        if pair_id in bayes_factors_db.keys():
            bayes_factor = bayes_factors_db.get(pair_id)
            log_print(["Redis GET bayes_factors_db pair:", pair_id])
            if model_a_id < model_b_id:
                return bayes_factor
            else:
                return (1.0/bayes_factor)
    else:
        
        model_a = QML.modelClassForRemoteBayesFactor(modelID=model_a_id,
            host_name=host_name, port_number=port_number, qid=qid, 
            log_file=log_file
        )
        model_b = QML.modelClassForRemoteBayesFactor(modelID=model_b_id,
            host_name=host_name, port_number=port_number, qid=qid, 
            log_file=log_file
        )

        log_print(["Start"])
        first_t_idx = len(model_a.Times) - num_times_to_use
        if num_times_to_use == 'all' or len(model_a.Times) < num_times_to_use:
            times_a = model_a.Times
            times_b = model_b.Times
        else:
            times_a = model_a.Times[first_t_idx:]
            times_b = model_b.Times[first_t_idx:]
        
        log_print(["Computing log likelihoods."])
        #print("Num times to use", num_times_to_use)
        #print("Model", model_a.ModelID, " Times:", times_a)
        #print("Model", model_b.ModelID, " Times:", times_b)
        log_l_a = log_likelihood(model_a, times_b)
        log_l_b = log_likelihood(model_b, times_a)     
        log_print(["Log likelihoods computed."])

        bayes_factor = np.exp(log_l_a - log_l_b)
        if bayes_factor < 1e-160:
            bayes_factor = 1e-160
        
        pair_id = DataBase.unique_model_pair_identifier(model_a_id, model_b_id)
        print("Bayes Factor:", pair_id)
        if float(model_a_id) < float(model_b_id):
            # so that BF in db always refers to (a/b), not (b/a). 
            bayes_factors_db.set(pair_id, bayes_factor)
            log_print(["Redis SET bayes_factors_db, pair:", pair_id,
                "bayes:", bayes_factor]
            )
        else:
            bayes_factors_db.set(pair_id, (1.0/bayes_factor))
            log_print(["Redis SET bayes_factors_db, pair:", pair_id, 
                "bayes:", (1.0/bayes_factor)]
            )

        if bayes_factor > bayes_threshold: 
            bayes_factors_winners_db.set(pair_id, 'a')
            log_print(["Redis SET bayes_factors_winners_db, pair:", 
                pair_id, "winner: a"]
            )
        elif bayes_factor < (1.0/bayes_threshold):
            bayes_factors_winners_db.set(pair_id, 'b')
            log_print(["Redis SET bayes_factors_winners_db, pair:", 
                pair_id, "winner: b"]
            )
        else:
            log_print(["Neither model much better."])

        
        if branchID is not None:    
            # only want to fill these lists when comparing models within branch
            log_print(["Redis INCR active_branches_bayes branch:", branchID])
            active_branches_bayes.incr(int(branchID), 1)  
        else:
            active_interbranch_bayes.set(pair_id, True)
            log_print(["Redis SET active_interbranch_bayes pair:", pair_id, 
                "; set:True"]
            )
        time_end = time.time()
        log_print(["Finished. rq time: ", str(time_end-time_start)])
    
        return bayes_factor
    
    
def log_likelihood(model, times):
    updater = model.Updater
    sum_data = 0
    #print("log likelihood function. Model", model.ModelID, "\n Times:", times)

    for i in range(len(times)):
        exp = get_exp(model, [times[i]])
    #    print("exp:", exp)
        params_array = np.array([[model.TrueParams[0]]]) # TODO this will cause an error for multiple parameters
        datum = updater.model.simulate_experiment(params_array, exp, repeat=1)
        sum_data+=datum       
        updater.update(datum, exp)

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
