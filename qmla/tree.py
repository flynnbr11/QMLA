from __future__ import absolute_import
from __future__ import print_function 

import math
import numpy as np
import os as os
import sys as sys
import pandas as pd
import time as time
from time import sleep
import random

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle

pickle.HIGHEST_PROTOCOL = 4  # TODO if >python3, can use higher protocol
plt.switch_backend('agg')

class qmla_tree():
    r"""
    Tree corresponding to a growth rule for management within QMLA.

    """

    def __init__(
        self, 
        growth_class
    ):
        self.growth_class = growth_class
        self.branches = {}
        self.models = {}
        self.parent_to_child_relationships = {}
        
        self.completed = self.growth_class.tree_completed_initially
        self.initial_models = self.growth_class.initial_models

    def get_branch_champions(self):
        return None


class qmla_branch():
    def __init__(
        self,
        branch_id, 
        models, # dictionary {id : name} 
        tree
    ):
        self.tree = tree # qmla_tree instance
        self.growth_class = self.tree.growth_class
        self.growth_rule = self.growth_class.growth_generation_rule
        self.bayes_points = {}
        self.rankings = {}

        self.models_by_id = models
        self.resident_models = list(self.models_by_id.values())
        self.resident_model_ids = sorted(self.models_by_id.keys())
        self.num_models = len(self.resident_models)
        self.num_model_pairs = num_pairs_in_list(self.num_models)

        self.model_learning_complete = False
        self.comparisons_completed = False

        # To set during QMLA
        self.precomputed_models = []
        self.num_precomputed_models = 0




    def get_champion(self):
        return None


class qmla_model():
    def __init__(
        self, 
        model_id,
        model_name,
    ):
        self.model_id = model_id
        self.model_name = model_name
        self.resident_on_branches = []
        self.resident_on_trees = []



def num_pairs_in_list(num_models):
    if num_models <= 1:
        return 0

    n = num_models
    k = 2  # ie. nCk where k=2 since we want pairs

    try:
        a = math.factorial(n) / math.factorial(k)
        b = math.factorial(n - k)
    except BaseException:
        print("Numbers too large to compute number pairs. n=", n, "\t k=", k)

    return a / b
