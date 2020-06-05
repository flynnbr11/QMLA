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

pickle.HIGHEST_PROTOCOL = 4
plt.switch_backend('agg')

class GrowthRuleTree():
    r"""
    Tree corresponding to a growth rule for management within QMLA.

    Each :class:`~qmla.growth_rules.GrowthRule` in the  :class:`~qmla.QuantumModelLearningAgent` 
    instance is associated with one tree (instance of this class).
    The tree interacts with the QMLA environment, for instance to choose the next set of models 
    to learn through :meth:`~qmla.GrowthRuleTree.next_layer`. 
    :class:`~qmla.BranchQMLA` exist both on the tree and the QMLA environment. 

    :param GrowthRule growth_class: instance of the growth class to associate with this tree.
    """

    def __init__(
        self, 
        growth_class,
    ):
        self.growth_class = growth_class
        self.growth_class.tree = self

        self.growth_rule = self.growth_class.growth_generation_rule
        self.log_file = self.growth_class.log_file
        self.log_print([ "Tree started for {}".format(self.growth_rule) ])

        # Infrastructure
        self.branches = {}
        self.models = {}
        self.model_instances = {}
        self.parent_children = {}
        self.child_parents = {}        
        self.spawn_step = 0
        self.prune_step = 0 

    ##########
    # Section: Interface with growth rule
    ##########

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

        if self.growth_class.branch_comparison_strategy == 'all':
            pairs_to_compare = 'all'
        elif self.growth_class.branch_comparison_strategy == 'optimal_graph':
            pairs_to_compare, graph = qmla.shared_functionality.model_pairing_strategies.find_efficient_comparison_pairs(
                model_names = self.initial_models
            )
            self.log_print(["Using optimal graph to select subset of model pairs to compare. ({} pairs)".format(len(pairs_to_compare))])
        else: 
            pairs_to_compare = 'all'


        return self.initial_models, pairs_to_compare

    def next_layer(
        self, 
        **kwargs
    ):
        r"""
        Determine the next set of models, for the next branch of this growth rule tree. 

        These are generated iteratively in stages, until that stage is marked as complete within the growth rule.
        * Spwaning: using the set of models from the previous branch to construct new models, 
          via :meth:`~qmla.growth_rules.GrowthRule.generate_models`. 
        * The spawning stage is terminated when :meth:`~qmla.growth_rules.GrowthRule.check_tree_completed`
          returns True. For example, if a predetermined number of spawn steps have passed, 
          or if the model learning has converged. 
        * After the spawn stage, the tree enters the pruning stage. 
          For example, parental collapse: branch champions of this tree
          are compared with their parents; if either parent or child is strongly superior, 
          the loser does not progress to the next branch. 
        * Pruning is carried out by :meth:`~qmla.growth_rules.GrowthRule.tree_pruning`, until
          :meth:`~qmla.growth_rules.GrowthRule.check_tree_pruned returns True.
        * After the pruning stage, it must be possible for this tree to nominate a list of 
          models to be considered for global champion, by :meth:`~qmla.GrowthRuleTree.nominate_champion`. 

        The set of models decided by this method are placed on the next QMLA branch. 
        ``pairs_to_compare`` here specifies which pairs of models on that next branch should be compared. 
        Typically, model learning corresponds to spawning; during this stage all models on branches 
        should be compared. 
        Pruning then corresponds to comparing models which have already been leared 
        (though new unlearned models will be learned). 
        During pruning, it may not be necessary to compare all pairs, for instance if it is desired only 
        to compare parent/child pairs at a given branch. 

        :param dict kwargs: key word args passed directly to spawning/pruning functions. 
        :return list model_list: list of model names to place on the next QMLA branch
        :return str or list pairs_to_compare: which model IDs within the model_list
            to perform comparisons between. 
        """

        # record end of previous generation
        # self.growth_class.ratings_class.record_current_ratings(
        #     model_list = , generation = self.spawn_step+0.85, marker='end'
        # )

        if not self.growth_class.check_tree_completed(spawn_step = self.spawn_step):
            self.spawn_step += 1
            self.log_print(["Next layer - spawn"])
            model_list =  self.growth_class.generate_models(
                spawn_step = self.spawn_step,             
                **kwargs
            )
            if self.growth_class.branch_comparison_strategy == 'all':
                pairs_to_compare = 'all'
            elif self.growth_class.branch_comparison_strategy == 'optimal_graph':
                pairs_to_compare, graph = qmla.shared_functionality.model_pairing_strategies.find_efficient_comparison_pairs(
                    model_names = model_list
                )
                self.log_print(["Using optimal graph to select subset of model pairs to compare. ({} pairs)".format(len(pairs_to_compare))])
            else: 
                pairs_to_compare = 'all'

        elif not self.growth_class.check_tree_pruned(prune_step = self.prune_step):
            self.prune_step += 1
            self.log_print(["Next layer - prune"])
            model_list, pairs_to_compare = self.growth_class.tree_pruning(
                previous_prune_branch = kwargs['called_by_branch']
            )
        else:
            self.log_print([
                "Trying to generate next layer but neither pruning or spawning."
            ])

        model_list = list(set(model_list))
        model_list = [qmla.database_framework.alph(mod) for mod in model_list]
        return model_list, pairs_to_compare


    def finalise_tree(self, **kwargs):
        r""" 
        After learning/pruning complete, run :meth:`~qmla.GrowthRule.finalise_model_learning`. 
        """

        last_branch = self.branches[max(self.branches.keys())]
        kwargs['evaluation_log_likelihoods'] = last_branch.evaluation_log_likelihoods
        kwargs['branch_model_points'] = last_branch.bayes_points

        self.growth_class.finalise_model_learning(
            **kwargs
        )

    def nominate_champions(
        self,
    ):
        r"""
        After tree has completed, nominate champion(s) to contest for global champion. 

        Growth rules can nominate one or more models as global champion. 
        The nominated champions of all GRs enter an all-vs-all contest
        in :meth:`~qmla.QuantumModelLearningAgent.compare_nominated_champions`, 
        with the winner deemed the global champion. 
        This tree object interfaces with the GR, so here it is just a wrapper for 
        :meth:`~qmla.growth_rules.GrowthRule.nominate_champions`.
        """
        return self.growth_class.nominate_champions()

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
        r"""
        Add a branch to this tree. 

        Generate a new :class:`~qmla.BranchQMLA`, places the given models on it, 
        and assign it to this tree. Return the object, so it can also act as a branch/layer in
        the :class:`~qmla.QuantumModelLearningAgent` environment. 
        Note models can reside on multiple branches. 

        :param int branch_id: branch ID unique to this branch and new in the QMLA environment.
        :param list models: model names to place on this branch.
        :param list pairs_to_compare: list of pairs of models to compare on this branch.
        :param dict model_instances: :class:`~qmla.ModelInstanceForStorage` instances of 
            each model id to be placed on this branch.
        :param list precomputed_models: models to place on branch
            which have already been learned on a previous branch.
        :param int spawning_branch: branch ID spawning this branch, to be recorded 
            as the new branch's parent. 
        :return BranchQMLA branch: instance of branch
        """
        
        # Instantiate branch
        branch = BranchQMLA(
            branch_id = branch_id, 
            models = models, 
            model_instances = model_instances, 
            precomputed_models = precomputed_models,
            tree = self, 
            spawning_branch = spawning_branch,
            pairs_to_compare = pairs_to_compare, 
            **kwargs            
        )

        # Update trees records
        self.branches[branch_id] = branch
        for m in model_instances:
            self.model_instances[m] = model_instances[m]
        
        # Record current state of ratings
        model_ids = list(model_instances.keys())
        # self.growth_class.ratings_class.record_current_ratings(
        #     model_list = model_ids, generation = self.spawn_step, marker='start'
        # )

        return branch

    ##########
    # Section: Inspect tree
    ##########

    def is_tree_complete(
        self,
    ):
        r""" 
        Check if tree has completed and doesn't need any further branches. 
        """

        # Check both spawning and pruning stages complete.
        tree_complete = (
            self.growth_class.check_tree_completed(spawn_step = self.spawn_step)
            and
            self.growth_class.check_tree_pruned(prune_step = self.prune_step)
        )
        
        # If initially complete, don't need any further iterations.
        if self.growth_class.tree_completed_initially:
            self.log_print(["Tree complete initially."])
            tree_complete = True

        self.log_print(["Checking if tree complete... ", tree_complete])

        return tree_complete

    ##########
    # Section: Utilities
    ##########

    def log_print(self, to_print_list):
        r"""Wrapper for :func:`~qmla.print_to_log`"""
        qmla.logging.print_to_log(
            to_print_list = to_print_list,
            log_file = self.log_file,
            log_identifier = 'Tree {}'.format(self.growth_rule)
        )

class BranchQMLA():
    def __init__(
        self,
        branch_id, 
        models, 
        model_instances,
        pairs_to_compare, 
        tree,
        precomputed_models,
        spawning_branch,
    ):
        r"""
        Branch on which to group models. 

        Branches simultenously belong to a :class:`~qmla.GrowthRuleTree` and the main
        :class:`~qmla.QuantumModelLearningAgent` environment, with the same ID on both. 
        Sets of models reside on each branch. Models can reside on multiple branches. 
        Models are instances of :class:`~qmla.ModelInstanceForStorage`. 
        Branches are essentially a mechanism to group models within a growth rule.
        By default, all models on a branch are compared with each other, but it is 
        possible to compare only a subset, e.g. during the tree pruning stage. 
        Models on branches are learned via 
        :meth:`~qmla.QuantumModelLearningAgent.learn_models_on_given_branch`; 
        if a model was previously considered on another branch it is not learned again. 
        Model comparisons for the branch are performed by 
        :meth:`~qmla.QuantumModelLearningAgent.compare_models_within_branch`. 

        :param int branch_id: branch ID unique to this branch and new in the QMLA environment.
        :param list models: model names to place on this branch.
        :param dict model_instances: :class:`~qmla.ModelInstanceForStorage` instances of 
            each model id to be placed on this branch.
        :param list pairs_to_compare: list of pairs of models to compare on this branch.
        :param GrowthRuleTree tree: tree to which this branch is associated.
        :param list precomputed_models: models to place on branch
            which have already been learned on a previous branch.
        :param int spawning_branch: branch ID spawning this branch, to be recorded 
            as the new branch's parent. 
        """

        self.branch_id = branch_id

        # Get attributes from tree of this branch
        self.tree = tree 
        self.log_file = self.tree.log_file
        self.growth_class = self.tree.growth_class
        self.growth_rule = self.growth_class.growth_generation_rule

        self.log_print([
            "Branch {} on tree {}".format(self.branch_id, self.tree),
        ])

        # Get parent branch of this branch
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

        # Models to place on the branch
        # TODO tidy up how models passed etc, a bit redundant
        self.model_instances = model_instances
        self.models = models
        self.models_by_id = models
        self.resident_model_ids = sorted(self.models_by_id.keys())
        self.resident_models = list(self.models_by_id.values())
        self.num_models = len(self.resident_models)

        # Choose models to compare within this branch
        if pairs_to_compare == 'all':
            self.log_print([
                "All pairs to be compared on branch ", self.branch_id
            ])
            self.pairs_to_compare = list(
                itertools.combinations(
                    self.resident_model_ids, 2
                )
            )
        else:
            self.log_print([
                "Comparison pairs passed:", pairs_to_compare
            ])
            self.pairs_to_compare = pairs_to_compare
        self.num_model_pairs = len(self.pairs_to_compare)
        
        # Models already considered on a previous branch
        self.precomputed_models = precomputed_models
        self.num_precomputed_models = len(self.precomputed_models)
        
        self.unlearned_models = list(
            set(self.resident_models) - set(self.precomputed_models)
        )

        # Interface with QMLA: attributes tp be called/edited continuously
        self.model_learning_complete = False
        self.comparisons_complete = False
        self.bayes_points = {}
        self.rankings = [] # ordered from best to worst

        self.log_print([
            "New branch {}; models: {}".format(
                self.branch_id, 
                self.models
            )
        ])

    ##########
    # Section: Utilities
    ##########

    def log_print(self, to_print_list):
        qmla.logging.print_to_log(
            to_print_list = to_print_list,
            log_file = self.log_file,
            log_identifier = 'Branch {}'.format(self.branch_id)
        )


