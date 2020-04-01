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

import qmla.logging 

pickle.HIGHEST_PROTOCOL = 4  # TODO if >python3, can use higher protocol
plt.switch_backend('agg')

class qmla_tree():
    r"""
    Tree corresponding to a growth rule for management within QMLA.

    """

    def __init__(
        self, 
        growth_class,
        log_file
    ):
        self.growth_class = growth_class
        self.growth_rule = self.growth_class.growth_generation_rule
        self.log_file = log_file
        
        self.branches = {}
        self.models = {}
        self.parent_to_child_relationships = {}
        
        self.completed = self.growth_class.tree_completed_initially
        self.spawn_step = 0
        self.initial_models = self.growth_class.initial_models

        self.branch_champions = {}
        self.branch_champions_by_dimension = {}

        self.ghost_branches = {}
        self.ghost_branch_list = []

    def spawn_models(
        self, 
        model_list, 
        **kwargs
    ):
        return self.growth_class.generate_models(
            model_list = model_list, 
            **kwargs
        )

    def new_branch(
        self, 
        branch_id, 
        models, 
        precomputed_models,
        **kwargs
    ):
        branch = qmla_branch(
            branch_id = branch_id, 
            models = models, 
            precomputed_models = precomputed_models,
            tree = self, # TODO is this safe??
            **kwargs            
        )
        self.branches[branch_id] = branch
        return branch

    def get_branch_champions(self):      
        all_branch_champions = [
            branch.get_champion() 
            for branch in self.branches
        ]
       
        return all_branch_champions

    def log_print(self, to_print_list):
        qmla.logging.print_to_log(
            to_print_list = to_print_list,
            log_file = self.log_file,
            log_identifier = 'Tree {}'.format(self.growth_rule)
        )

    def is_tree_complete(
        self,
        spawn_step=None
    ):
        tree_complete = self.growth_class.check_tree_completed(spawn_step = self.spawn_step)
        self.log_print(
            [
                "Checking if tree complete... ", 
                tree_complete
            ]
        )
        return tree_complete



class qmla_branch():
    def __init__(
        self,
        branch_id, 
        models, # dictionary {id : name} 
        tree,
        precomputed_models
    ):
        # housekeeping
        self.branch_id = branch_id
        self.tree = tree # qmla_tree instance
        self.log_file = self.tree.log_file
        self.growth_class = self.tree.growth_class
        self.growth_rule = self.growth_class.growth_generation_rule
        self.spawn_step = 0 
        self.log_print([
            "Branch {} has tree {}".format(self.branch_id, self.tree)
        ])
        self.models = models
        self.models_by_id = models
        self.resident_models = list(self.models_by_id.values())
        self.resident_model_ids = sorted(self.models_by_id.keys())
        self.num_models = len(self.resident_models)
        self.num_model_pairs = num_pairs_in_list(self.num_models)

        self.precomputed_models = precomputed_models
        self.num_precomputed_models = len(self.precomputed_models)
        
        self.unlearned_models = list(
            set(self.resident_models)
            - set(self.precomputed_models)
        )
        if self.num_precomputed_models == 0:
            self.is_ghost_branch = True
        else:
            self.is_ghost_branch = False

        self.log_print(
            [
                "New branch {}; models: {}".format(
                    self.branch_id, 
                    self.models
                )
            ]
        )

        # To be called/edited continusously by QMLA
        self.model_learning_complete = False
        self.comparisons_complete = False
        self.bayes_points = {}
        self.rankings = [] # ordered from best to worst

    def get_champion(self):
        self.champion = self.rankings[0]
        return self.champion


    def log_print(self, to_print_list):
        qmla.logging.print_to_log(
            to_print_list = to_print_list,
            log_file = self.log_file,
            log_identifier = 'Branch {}'.format(self.branch_id)
        )



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
