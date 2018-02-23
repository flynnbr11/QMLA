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


def get_directory_name_by_time(just_date=False):
    import datetime
    # Directory name based on date and time it was generated 
    # from https://www.saltycrane.com/blog/2008/06/how-to-get-current-date-and-time-in/
    now =  datetime.date.today()
    year = now.strftime("%y")
    month = now.strftime("%b")
    day = now.strftime("%d")
    hour = datetime.datetime.now().hour
    minute = datetime.datetime.now().minute
    date = str (str(day)+'_'+str(month)+'_'+str(year) )
    time = str(str(hour)+'_'+str(minute))
    name = str(date+'/'+time+'/')
    if just_date is False:
        return name
    else: 
        return str(date+'/')

global_true_op = 'x'
true_params  = [0.5]
num_part = 10
num_exp=5
best_resample_threshold=0.5
best_resampler_a = 0.9
best_pgh = 1.0


def create_and_run_qmd(
          qmd_id, 
          pickle_qmd_class = False,
          initial_op_list=[global_true_op], 
          true_operator=global_true_op, 
          true_param_list=true_params, 
          num_particles=num_part,
          qle=True,
          max_num_branches = 0,
          max_num_qubits = 2, 
          resample_threshold = best_resample_threshold,
          resampler_a = best_resampler_a,
          pgh_prefactor = best_pgh
        ):  
          print("Inside create_and_run_qmd function.")
          initial_op_list = true_params
          qmd = QMD(num_particles=10, 
                    max_num_qubits=2
          )
          # qmd.runAllActiveModelsIQLE(num_exp=num_exp)
          qmd.runQMD(num_exp = num_exp, spawn=False)

          print("\nQMD Test ", str(qmd_id))
          print("QMD Champion:", qmd.ChampionName)
          
          return qmd
          del qmd

