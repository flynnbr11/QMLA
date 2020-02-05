from __future__ import print_function  # so print doesn't show brackets

import numpy as np
import itertools as itr
import warnings
import os as os
import sys as sys
import pandas as pd
import warnings
import copy
import time as time

from fermilib.ops import FermionOperator
from fermilib.transforms import get_sparse_operator
import fermilib

import qmla.database_framework as database_framework
import qmla.model_naming as model_naming
import qmla.logging

"""
Essential functions. Functions below are specific, for generating terms according to given rules. new_model_list is the wrapper function which calls the selected generator rule.
"""

def log_print(
    to_print_list, 
    log_file
):
    qmla.logging.print_to_log(
        to_print_list = to_print_list, 
        log_file = log_file,
        log_identifier = 'Model Generation'
    )

"""
File kept in case some model generation functions can be reused, 
in which case they can be stored here. 
"""