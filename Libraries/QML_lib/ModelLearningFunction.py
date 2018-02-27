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

import redis
redis_db = redis.StrictRedis(host="localhost", port=6379, db=0)
learned_models_info = redis.StrictRedis(host="localhost", port=6379, db=1)
  
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

def learnModelRemote(name, modelID, qmd_info, remote=False):

        # Get params from qmd_info
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
        probe_dict = qmd_info['probe_dict']
        sigma_threshold = qmd_info['sigma_threshold']
        
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
        qml_instance.InitialiseNewModel(
          trueoplist = true_ops,
          modeltrueparams = true_params,
          simoplist = op.constituents_operators,
          simparams = [sim_pars],
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
        
        qml_instance.UpdateModel(n_experiments = num_experiments, sigma_threshold = sigma_threshold)
        updated_model_info = copy.deepcopy(qml_instance.learned_info_dict()) # possibly need to take a copy
        del qml_instance

        if remote: 
#            pickled_file = open('test_pickled_model_dict.pkl', "wb")
#            pickle.dump(learned_model_info, pickled_file, protocol=2)
#            pickled_file.close()
            compressed_info = pickle.dumps(updated_model_info)
            learned_models_info.set(modelID, compressed_info)
            redis_db.set(modelID, True)

            del updated_model_info
            del compressed_info
            print("Model", name, "learned and pickled to redis DB.")
            return None
        else: 
            return updated_model_info
