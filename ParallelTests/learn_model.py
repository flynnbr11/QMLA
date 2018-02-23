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

sys.path.append(os.path.join("..", "Libraries","QML_lib"))
import Evo as evo
import DataBase 
from QMD import QMD #  class moved to QMD in Library
import QML
import ModelGeneration 
import BayesF
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
paulis = ['x', 'y', 'z'] # will be chosen at random. or uncomment below and comment within loop to hard-set

import time as time 
import argparse
parser = argparse.ArgumentParser(description='Pass variables for (I)QLE.')

import redis
import pickle
rds = redis.StrictRedis(host='localhost', port=6379, db=0)

def LearnModel(name, num_particles, num_exp):
  print("In learn model function")
  op= DataBase.operator(name)
  true_ops = op.constituents_operators
  sim_ops = op.constituents_operators
  true_params = [0.5]
  sim_params = [true_params]
  print("initialising model")
  mod = QML.ModelLearningClass(name=name)
  mod.InitialiseNewModel(
          trueoplist = true_ops,
          modeltrueparams = true_params,
          simoplist = sim_ops,
          simparams = sim_params,
          numparticles = num_particles,
          use_exp_custom = True,
          enable_sparse = True,
          modelID = 1,
          resample_thresh = 0.5,
          resampler_a = 0.9,
          pgh_prefactor = 1.0,
          gaussian=False,
          debug_directory = None,
          qle = True
        )
        
  print("About to update model")
  mod.UpdateModel(n_experiments = num_exp)

  print("Saving pickled objects to redis DB")
  redis_db_name = str(name)+'_learned'
  redis_updater_name = str(name)+'_updater'

  #rds.set(redis_db_name, pickle.dumps(mod))
#  rds.set(redis_updater_name, pickle.dumps(mod.Updater))
  p = rds.pubsub()
  print("Sending message on channel ", redis_db_name)

  rds.publish(redis_db_name, {"message": 'model_finished'})
  
  rds.publish('test_channel', {'message' : 'test message from new channel'})
#  p.publish(redis_db_name, {"message": 'model_finished'})
  

  
  print("Model ", name, " learned and pickled.")
    
