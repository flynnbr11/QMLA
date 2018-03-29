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

## Single function call, given QMDInfo and a name, to learn model entirely. 

def learnModelRemote(name, modelID, branchID, qmd_info=None, remote=False):
        print("QHL for", name, "remote:", remote)
        # Get params from qmd_info
        if qmd_info == None:
            print("Trying to load qmd info from redis db")
            qmd_info = pickle.loads(qmd_info_db['QMDInfo'])
            print("Trying to load probe dict info from redis db")
            probe_dict = pickle.loads(qmd_info_db['ProbeDict'])
        else: # if in serial, qmd_info given, with probe_dict included in it. 
            probe_dict = qmd_info['probe_dict']
        print("QMD info loaded")

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
        qml_instance = ModelLearningClass(name=name, num_probes = num_probes, probe_dict=probe_dict)

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
        print("Trying to initialise new model with id", modelID, " and name", name)
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
          qle = qle
        )
        print("QML instance generated")
        qml_instance.UpdateModel(n_experiments = num_experiments, sigma_threshold = sigma_threshold)
        updated_model_info = copy.deepcopy(qml_instance.learned_info_dict()) # possibly need to take a copy
        del qml_instance

        compressed_info = pickle.dumps(updated_model_info, protocol=2) #TODO is there a way to use higher protocol when using python3 for faster pickling?
        learned_models_info.set(str(modelID), compressed_info)
        learned_models_ids.set(str(modelID), True)

        current = int(active_branches_learning_models.get(int(branchID))) # if first to finish
        active_branches_learning_models.set(int(branchID), current+1)    
            
        if remote: 
            del updated_model_info
            del compressed_info
            print("Model", name, "learned and pickled to redis DB.")
            return None
        else: 
            return updated_model_info
