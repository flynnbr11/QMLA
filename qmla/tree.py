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
import itertools

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pickle

import qmla.logging 
import qmla.database_framework

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
        self.growth_class.tree = self
        self.log_file = log_file
        self.log_print([
            "Tree started for {}".format(self.growth_rule)
        ])
        self.branches = {}
        self.models = {}
        self.model_instances = {}
        self.parent_children = {}
        self.child_parents = {}
        
        self.completed = self.growth_class.tree_completed_initially
        self.prune_complete = False
        self.prune_branches = []
        self.spawn_step = 0
        self.prune_step = 0
        self.initial_models = self.growth_class.initial_models

        self.branch_champions = {}
        self.branch_champions_by_dimension = {}

        self.ghost_branches = {}
        self.ghost_branch_list = []

    def get_initial_models(
        self, 
    ):
        if self.growth_class.initial_models is None:
            self.log_print([
                "Initial models not set; retrieving from generate_models"
            ])
            self.initial_models = self.growth_class.generate_models(
                model_list=None
            )
        else:
            self.initial_models = self.growth_class.initial_models
        return self.initial_models



    def next_layer(
        self, 
        **kwargs
    ):
        if not self.growth_class.check_tree_completed(spawn_step = self.spawn_step):
            self.log_print([
                "Next layer - spawn"
            ])

            self.spawn_step += 1
            model_list =  self.growth_class.generate_models(
                spawn_step = self.spawn_step,             
                **kwargs
            )
            pairs_to_compare = 'all'

        elif not self.growth_class.prune_complete:
            self.log_print([
                "Next layer - prune"
            ])
            model_list, pairs_to_compare = self.growth_class.tree_pruning(
                previous_prune_branch = kwargs['called_by_branch']
            )

        model_list = list(set(model_list))
        model_list = [qmla.database_framework.alph(mod) for mod in model_list]
        return model_list, pairs_to_compare

    # def spawn_models(
    #     self, 
    #     **kwargs
    # ):
    #     self.spawn_step += 1
    #     new_models =  self.growth_class.generate_models(
    #         spawn_step = self.spawn_step,             
    #         **kwargs
    #     )
    #     new_models = list(set(new_models))
    #     new_models = [qmla.database_framework.alph(mod) for mod in new_models]
    #     pairs_to_compare = 'all'

    #     return new_models, pairs_to_compare

    # def prune_tree(
    #     self, 
    #     previous_prune_branch = 0, 
    #     **kwargs
    # ):
    #     self.prune_step += 1

    #     return self.growth_class.tree_pruning(
    #         previous_prune_branch = previous_prune_branch,
    #         prune_step = self.prune_step
    #     )

    def nominate_champions(
        self,
    ):
        if len(self.prune_branches) > 0:
            final_branch = self.branches[
                self.prune_branches[-1]
            ]
        else:
            self.log_print([
                "Prune branches not listed; assuming final branch to hold tree champion"
            ])
            branches = sorted(list(self.branches.keys()))
            final_branch = self.branches[
                max(branches)
            ]

        self.log_print([
            "Final branch is {}".format(final_branch.branch_id)
        ])

        return [final_branch.champion_name]

    def new_branch_on_tree(
        self, 
        branch_id, 
        models, 
        pairs_to_compare,
        model_instances,
        precomputed_models,
        spawning_branch, 
        **kwargs
    ):
        for m in model_instances:
            self.model_instances[m] = model_instances[m]
        
        branch = qmla_branch(
            branch_id = branch_id, 
            models = models, 
            model_instances = model_instances, 
            precomputed_models = precomputed_models,
            tree = self, # TODO is this safe??
            spawning_branch = spawning_branch,
            pairs_to_compare = pairs_to_compare, 
            **kwargs            
        )
        self.branches[branch_id] = branch
        return branch

    def get_branch_champions(self):      
        all_branch_champions = [
            branch.champion_name
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
    ):
        tree_complete = (
            self.growth_class.check_tree_completed(spawn_step = self.spawn_step)
            and
            self.growth_class.prune_complete
        )
        if self.growth_class.tree_completed_initially:
            self.log_print([
                "Tree complete initially."
            ])
            tree_complete = True
        self.log_print(
            [
                "Checking if tree complete... ", 
                tree_complete
            ]
        )
        if tree_complete:
            self.log_print([
                "Complete at spawn step", self.spawn_step
            ])
        return tree_complete

    def is_pruning_complete(self):
        # TODO do we need calls to this in QMLA? or just check prune_complete?
        return self.growth_class.prune_complete


class qmla_branch():
    def __init__(
        self,
        branch_id, 
        models, # dictionary {id : name} 
        model_instances, # dictionary {id : ModelInstanceForStorage} 
        tree,
        precomputed_models,
        spawning_branch,
        pairs_to_compare, 
    ):
        # housekeeping
        self.branch_id = branch_id
        self.tree = tree # qmla_tree instance
        self.log_file = self.tree.log_file
        self.growth_class = self.tree.growth_class
        self.growth_rule = self.growth_class.growth_generation_rule
        self.log_print([
            "QMLA Branch object for {}. spawning branch:{}".format(
                branch_id, 
                spawning_branch
            )
        ])
        try:
            self.parent_branch = self.tree.branches[spawning_branch]
            self.log_print([
                "Setting parent branch of {} -> parent is {}".format(
                    self.branch_id, 
                    self.parent_branch.branch_id
                )
            ])
        except:
            self.parent_branch = None
            self.log_print(["Failed to set parent branch for ", branch_id])
        self.prune_branch = False
        self.spawn_step = 0 
        self.log_print([
            "Branch {} has tree {}".format(self.branch_id, self.tree)
        ])
        self.model_instances = model_instances
        self.models = models
        self.models_by_id = models
        self.resident_model_ids = sorted(self.models_by_id.keys())
        self.resident_models = list(self.models_by_id.values())
        self.num_models = len(self.resident_models)
        if pairs_to_compare == 'all':
            self.log_print([
                "All pairs to be compared on branch ", self.branch_id
            ])
            self.pairs_to_compare = list(
                itertools.combinations(
                    self.resident_model_ids, 
                    2
                )
            )
        else:
            self.pairs_to_compare = pairs_to_compare
        self.num_model_pairs = len(self.pairs_to_compare)
        self.model_parent_branch = {
            model_id : spawning_branch 
            for model_id in self.resident_model_ids
        }

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

        # To be called/edited continuously by QMLA
        self.model_learning_complete = False
        self.comparisons_complete = False
        self.bayes_points = {}
        self.rankings = [] # ordered from best to worst

    def get_champion(self):
        self.champion = self.rankings[0]
        return self.champion

    def update_comparisons(
        self, 
        models_poinits, 
    ):
        self.bayes_points = models_points

    def set_as_prune_branch(
        self, 
        pairs_to_compare,
    ):
        self.pairs_to_compare = pairs_to_compare
        self.num_model_pairs = len(self.pairs_to_compare)
        self.log_print([
            "Setting as prune branch with pairs to compare:", 
            self.pairs_to_compare
        ])
        self.prune_branch=True
        self.tree.prune_branches.append(self.branch_id)

    def log_print(self, to_print_list):
        qmla.logging.print_to_log(
            to_print_list = to_print_list,
            log_file = self.log_file,
            log_identifier = 'Branch {}'.format(self.branch_id)
        )

class iterative_improvement_pruning(qmla_tree):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prune_tree(
        self, 
        previous_prune_branch = 0, 
        **kwargs
    ):
        self.prune_complete = True
        branches = sorted(list(self.branches.keys()))
        final_branch = self.branches[
            max(branches)
        ]
        return [final_branch.champion_name], []

