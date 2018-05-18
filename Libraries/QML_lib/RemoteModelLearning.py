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
    import redis
    import RedisSettings as rds
    enforce_serial = False  
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


def time_seconds():
    import datetime
    now =  datetime.date.today()
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute
    second = datetime.datetime.now().second
    time = str(str(hour)+':'+str(minute)+':'+str(second))
    return time



## Single function call, given QMDInfo and a name, to learn model entirely. 

def learnModelRemote(name, modelID, branchID, qmd_info=None, remote=False, host_name='localhost', port_number=6379, qid=0, log_file='rq_output.log'):
        print("QHL for", name, "remote:", remote)
        time_start = time.time()
        # Get params from qmd_info
        rds_dbs = rds.databases_from_qmd_id(host_name, port_number, qid)
        qmd_info_db = rds_dbs['qmd_info_db'] 
        learned_models_info = rds_dbs['learned_models_info']
        learned_models_ids = rds_dbs['learned_models_ids']
        active_branches_learning_models = rds_dbs['active_branches_learning_models']

        write_log_file = open(log_file, 'a')
        def log_print(to_print):
            identifier = str(str(time_seconds()) + " [RQ Learn "+str(modelID)+"]")
            print(identifier, str(to_print), file=write_log_file, flush=True)


        if qmd_info == None:
            #print("Trying to load qmd info from redis db")
            qmd_info = pickle.loads(qmd_info_db['QMDInfo'])
            #print("Trying to load probe dict info from redis db")
            probe_dict = pickle.loads(qmd_info_db['ProbeDict'])
        else: # if in serial, qmd_info given, with probe_dict included in it. 
            probe_dict = qmd_info['probe_dict']
        #print("QMD info loaded")

        true_ops = qmd_info['true_oplist']
        true_params = qmd_info['true_params']
        num_particles = qmd_info['num_particles']
        num_experiments = qmd_info['num_experiments']
        resampler_threshold = qmd_info['resampler_thresh']
        resampler_a = qmd_info['resampler_a']
        pgh_prefactor = qmd_info['pgh_prefactor']
        debug_directory = qmd_info['debug_directory']
        qle = qmd_info['qle']
        num_probes = qmd_info['num_probes']
        sigma_threshold = qmd_info['sigma_threshold']
        gaussian = qmd_info['gaussian']
        
        # Generate model and learn
        op = DataBase.operator(name = name)
        qml_instance = ModelLearningClass(name=name, num_probes = num_probes, probe_dict=probe_dict, qid=qid, log_file=log_file, modelID=modelID)

        sim_pars = []
        num_pars = op.num_constituents
        if num_pars ==1 : #TODO Remove this fixing the prior
          normal_dist=NormalDistribution(mean=true_params[0], var=0.1)
        else:  
          normal_dist = MultiVariateNormalDistributionNocov(num_pars)
        
        for j in range(op.num_constituents):
          sim_pars.append(normal_dist.sample()[0,0])
          
        # add model_db_new_row to model_db and running_database
        # Note: do NOT use pd.df.append() as this copies total DB,
        # appends and returns copy.
        qml_instance.InitialiseNewModel(
          trueoplist = true_ops,
          modeltrueparams = true_params,
          simoplist = op.constituents_operators,
          simparams = [sim_pars],
          simopnames = op.constituents_names,
          numparticles = num_particles,
          use_exp_custom = True,
          enable_sparse=True,
          modelID = modelID,
          resample_thresh = resampler_threshold,
          resampler_a = resampler_a,
          pgh_prefactor = pgh_prefactor,
          gaussian=False,
          debug_directory = debug_directory,
          qle = qle,
          host_name = host_name,
          port_number = port_number, 
          qid=qid,
          log_file = log_file
        )
        
        qml_instance.UpdateModel(n_experiments = num_experiments, sigma_threshold = sigma_threshold)
        updated_model_info = copy.deepcopy(qml_instance.learned_info_dict()) # possibly need to take a copy
        del qml_instance

        compressed_info = pickle.dumps(updated_model_info, protocol=2) #TODO is there a way to use higher protocol when using python3 for faster pickling? this seems to need to be decoded using encoding='latin1'.... not entirely clear why this encoding is used
        learned_models_info.set(str(modelID), compressed_info)
        learned_models_ids.set(str(modelID), True)

        current = int(active_branches_learning_models.get(int(branchID))) # if first to finish
        active_branches_learning_models.set(int(branchID), current+1)    
        time_end = time.time()

            
        if remote: 
            del updated_model_info
            del compressed_info
            print("Model", name, "learned and pickled to redis DB.")
            log_print("Learned")
#            print("Time taken to learn model:", time_end - time_start)
            time_taken = "Time:"+str(time_end-time_start)
            log_print(time_taken)
            return None
        else: 
            return updated_model_info
