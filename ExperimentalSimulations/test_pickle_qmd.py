from __future__ import print_function # so print doesn't show brackets

import os as os
import numpy as np
import itertools as itr

import sys as sys 
import pandas as pd
import warnings
import time as time
import random
import pickle
pickle.HIGHEST_PROTOCOL = 2

sys.path.append(os.path.join("..", "Libraries","QML_lib"))

import GlobalVariables
global_variables = GlobalVariables.parse_cmd_line_args(sys.argv[1:])
os.environ["TEST_ENV"] = 'test'

import RedisSettings as rds

# Set up redis 
# rds.redis_start(global_variables.host_name, global_variables.port_number, global_variables.qmd_id)


import Evo as evo
import DataBase 
from QMD import QMD #  class moved to QMD in Library


test_class = QMD()
print("Class created")


pickle.dump(test_class, open('test_class.p', 'wb'))
print("pickled")
