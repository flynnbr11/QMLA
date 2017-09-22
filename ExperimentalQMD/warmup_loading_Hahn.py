import qinfer as qi
#import qutip as qt
import numpy as np
import scipy as sp
from IPython.display import display, Math, Latex
from functools import partial
import matplotlib.pyplot as plt
import importlib as imp

import datetime
import os as os
import time as time

import logging as logging
import sys

print("Standard libraries imported")


import warnings as warnings
""" filter only the Weight Clipping warning """
#SET LEVEL WARNINGS TO 0 In FINAL VERSION
warnings.filterwarnings("ignore", message='Negative weights occured', category=RuntimeWarning)
print("Neg weights warning filtered")


# tell the user where files will be saved
dire = os.getcwd()
dire = dire+"/test_results/"
dire = os.path.normpath(dire)
print('Files will be saved in:')
print(dire)


# Add personalized library to PYTHONPATH
sys.path.append(os.path.join("..","Libraries","QML_lib"))
sys.path.append(os.path.join(".."))

import SETTINGS as SETTINGS # settings for the notebook
# SETTINGS.init()

from Norms import *
from EvalLoss import *

import ProbeStates as pros
import multiPGH as mpgh
import Evo as evo
import Distrib as distr
import HahnTheoModels as HTM
import IOfuncts as mIO
import Utils as uti


print("Customised libraries imported")

